

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedGemma 모델 QLoRA 기반 Fine-Tuning 스크립트 (v0.2, 멀티 이미지/샘플)

개요:
-----
본 스크립트는 MedGemma 모델에 대해 4-bit QLoRA 방식을 적용하여 파인튜닝을 수행합니다. 
단일 이미지뿐 아니라 복수 이미지 입력을 지원하며, 다중 GPU 환경에서의 안정성과 성능 개선을 중점으로 개선되었습니다.

주요 개선 사항:
---------------
1. 이미지 로딩 오류 방지를 위한 안전 로딩 (오류 시 검정 이미지 대체)
2. BLEU 및 METEOR 기반 평가 지원 (분산 실행 시 CLIP Score 제외)
3. dataloader_num_workers = 0 설정으로 다중 GPU 환경에서의 안정성 강화
4. 전처리 단계에서의 JSON 입력 및 이미지 입력 상태 확인 추가
5. JSON 내 불필요 공백 및 잘못된 텍스트 제거
6. 학습용(train)과 평가용(eval) 데이터셋 분리
7. accelerate 기반 멀티 GPU 학습 환경으로 전환 (단일 GPU 코드 → DDP/Accelerator로 통합)
8. 사용 예시 포함 (총 8개 GPU: NVIDIA RTX 6000 Ada 사용)


"""

import argparse
import datetime
import os
import random
import mimetypes
from typing import Any, Dict, List, Optional, Sequence, Tuple
from peft import get_peft_model, LoraConfig
import torch
from PIL import Image
from accelerate import Accelerator
from datasets import load_dataset
from evaluate import load as evload
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune MedGemma with QLoRA")

    parser.add_argument("--model_path", default="medgemma-4b-it", help="Path to the pretrained MedGemma model.")
    parser.add_argument("--train_json", required=True, help="JSON file containing training data with image paths and messages.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Steps between saving checkpoints.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Steps between logging loss.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers.")
    parser.add_argument("--image_size", type=int, default=896, help="Tile image size.")
    parser.add_argument("--no_vis", action="store_true", help="Skip visualizations and checks.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save final model")

    return parser.parse_args()

def print_trainable_parameters_custom(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"🔧 Trainable params: {trainable} | Total params: {total} | Trainable%: {100 * trainable / total:.4f}%")

def prepare_model_and_processor(
    model_path: str,
    lora_rank: int,
    accelerator: Accelerator,
) -> Tuple[Any, Any, List[str]]:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float32,
    )

    with accelerator.main_process_first():
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_disable()

    # Vision Tower gradient checkpointing
    vt = model.vision_tower
    vision_layers = (
        vt.encoder.layers
        if hasattr(vt, "encoder")
        else vt.vision_model.encoder.layers
    )
    for blk in vision_layers:
        blk.gradient_checkpointing = True

# LoRA 적용 대상 분리
    vision_linear_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and any(k in name.lower() for k in ["vision", "image", "vit", "visual"])
        ] 

    language_linear_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not any(k in name.lower() for k in ["vision", "image", "vit", "visual"])
        ]


    targets = vision_linear_modules + language_linear_modules

    print("🎯 LoRA will be applied to the following Linear modules:")
    print("\n🧠 Vision 관련 레이어:")
    if vision_linear_modules:
        for name in vision_linear_modules:
            print(f"  - {name}")
    else:
        print("  ❌ 없음")

    print("\n🧠 Language 관련 레이어:")
    for name in language_linear_modules:
        print(f"  - {name}")

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 4,
        lora_dropout=0.05,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    print("🧩 Trainable parameters summary after applying LoRA:")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print_trainable_parameters_custom(model)

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, targets



def load_and_split_dataset(
    json_path: str,
    accelerator: Accelerator,
) -> Tuple[Any, Any, Any]:
    def is_valid(example: Dict[str, Any]) -> bool:
        return (
            example.get("image")
            and os.path.exists(example["image"])
            and example.get("messages")
        )

    with accelerator.main_process_first():
        ds_all = load_dataset("json", data_files={"train": json_path})["train"].filter(is_valid)

    indices = list(range(len(ds_all)))
    random.seed(42)
    random.shuffle(indices)
    cut = int(len(indices) * 0.9)
    train_idx = indices[:cut]
    val_idx = indices[cut:] or indices[-1:]

    ds_train = ds_all.select(train_idx)
    ds_val = ds_all.select(val_idx)

    if accelerator.is_main_process:
        print(f"Total samples: {len(ds_all):,}")
        print(f"Train: {len(ds_train):,} | Val: {len(ds_val):,}")

    return ds_train, ds_val, ds_all

def load_image(path: str) -> Any:
    """
    Safely load an image or tensor from disk. If loading fails, return
    a black placeholder image instead of raising an exception.
    """
    mime, _ = mimetypes.guess_type(path)
    try:
        if mime and mime.startswith("image"):
            return Image.open(path).convert("RGB")
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Failed to load '{path}': {e}. Using fallback image.")
        return Image.new("RGB", (224, 224), color="black")

def collate_fn(
    batch: Sequence[Dict[str, Any]],
    processor: Any,
    ds_all: Any,
) -> Dict[str, Any]:
    def is_valid(example: Dict[str, Any]) -> bool:
        return (
            example.get("image")
            and os.path.exists(example["image"])
            and example.get("messages")
        )

    batch = [ex for ex in batch if is_valid(ex)] or [ds_all[0]]
    texts: List[str] = []
    images: List[List[Any]] = []
    for ex in batch:
        texts.append(
            processor.apply_chat_template(
                ex["messages"],
                add_generation_prompt=False,
                tokenize=False,
            ).strip()
        )
        images.append([load_image(ex["image"])])

    out = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = out["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    start_id = processor.tokenizer.convert_tokens_to_ids("<start_of_image>")
    soft_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    for i, ids in enumerate(out["input_ids"]):
        positions = (ids == start_id).nonzero(as_tuple=True)[0]
        for p in positions:
            q = p + 1
            while q < ids.size(0) and soft_id <= ids[q] < soft_id + 256:
                q += 1
            labels[i, p:q] = -100

    out["labels"] = labels
    return out

def load_metrics() -> Tuple[Any, Optional[Any]]:
    """
    Load evaluation metrics. Returns BLEU and METEOR (if available).
    CLIP score is omitted to avoid image loading in metrics.
    """
    bleu = evload("bleu")
    try:
        import nltk  # type: ignore
        nltk.download("wordnet", quiet=True)
        meteor = evload("meteor")
    except Exception:
        meteor = None
    return bleu, meteor

def compute_metrics_factory(
    processor: Any,
    bleu_metric: Any,
    meteor_metric: Optional[Any],
) -> Any:
    def compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        preds, labels = eval_pred
        preds_txt = processor.batch_decode(preds, skip_special_tokens=True)
        labels_txt = processor.batch_decode(labels, skip_special_tokens=True)
        results = {
            "bleu": bleu_metric.compute(
                predictions=preds_txt,
                references=[[t] for t in labels_txt],
            )["bleu"]
        }
        if meteor_metric is not None:
            results["meteor"] = meteor_metric.compute(
                predictions=preds_txt,
                references=[[t] for t in labels_txt],
            )["meteor"]
        return results
    return compute_metrics


from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

class PrintLossCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Step {state.global_step}] Training loss: {logs['loss']:.4f}")

class PrintEvalCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        print(f"\n[✅ Evaluation @ Step {state.global_step}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")



import datetime
import os
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
import torch

# ✅ 2. on_epoch_end 수정
class EvalOnEpochEndCallback(TrainerCallback):
    def __init__(self, val_dataset, processor, writer=None, max_samples=30):
        self.val_dataset = val_dataset
        self.processor = processor
        self.writer = writer or SummaryWriter()
        self.max_samples = max_samples

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n[EVAL] Epoch {state.epoch:.2f} - Generating BLEU score...")
        model.eval()
        scores = []

        for i, sample in enumerate(self.val_dataset):
            if i >= self.max_samples:
                break

            image = sample["image"]

            # ✅ flatten 적용
            try:
                flattened = flatten_messages(sample["messages"])
                if not flattened:
                    print(f"[WARN] 메시지 flatten 실패 → 건너뜀: {sample}")
                    continue

                gt_text = self.processor.apply_chat_template(
                    flattened,
                    add_generation_prompt=False,
                    tokenize=False
                ).strip()

                if not gt_text:
                    print(f"[WARN] 빈 gt_text 생성 → 건너뜀: {sample}")
                    continue

            except Exception as e:
                print(f"[ERROR] 메시지 처리 중 예외 발생: {e} → 건너뜀")
                continue

            # ✅ text 인자 포함하여 processor 호출
            try:
                inputs = self.processor(
                    text=[gt_text],
                    images=[load_image(image)],
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)

                pred_text = self.processor.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

                bleu_score = sentence_bleu([gt_text.split()], pred_text.split())
                scores.append(bleu_score)

            except Exception as e:
                print(f"[ERROR] 추론 중 예외 발생: {e} → 건너뜀")
                continue

        avg_bleu = sum(scores) / len(scores) if scores else 0.0
        print(f"[EVAL] Epoch {state.epoch:.2f} - BLEU score: {avg_bleu:.4f}")
        self.writer.add_scalar("eval/bleu", avg_bleu, int(state.epoch))




from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16")

    # 1. 모델 + LoRA 설정
    model, processor, used_lora_layers = prepare_model_and_processor(
        model_path=args.model_path,
        lora_rank=args.rank,
        accelerator=accelerator,
    )

    # 🔍 LoRA 파라미터 확인
    print("🧩 Trainable parameters summary after applying LoRA:")
    for layer in used_lora_layers:
        print(f"  - {layer}")

    # ✅ ViT 관련 레이어 포함 여부 확인
    vision_keywords = ["vision", "vit", "image", "visual"]
    has_vision_layer = any(any(k in layer_name.lower() for k in vision_keywords) for layer_name in used_lora_layers)

    if has_vision_layer:
        print("✅ LoRA가 Vision 관련 레이어에도 적용되었습니다.")
    else:
        print("❌ Vision 관련 레이어에는 LoRA가 적용되지 않았습니다.")

    # 2. 데이터셋 로딩
    ds_train, ds_val, ds_all = load_and_split_dataset(
        json_path=args.train_json,
        accelerator=accelerator,
    )

    # 3. 시각화 및 텍스트 검증
    if not args.no_vis and accelerator.is_main_process:
        unique_labels = set()
        for i in range(min(1000, len(ds_all))):
            for message in ds_all[i]["messages"]:
                if message.get("role") == "assistant":
                    for content in message.get("content", []):
                        text = content.get("text")
                        if text:
                            unique_labels.add(text.strip())
        label_count = len(unique_labels)
        print(f"📊 Unique labels: {label_count}")
        if label_count < 2:
            print("⚠️ Warning: low label diversity detected.")

        sample_batch = [ds_all[i] for i in range(min(4, len(ds_all)))]
        texts = [
            processor.apply_chat_template(
                s["messages"],
                add_generation_prompt=False,
                tokenize=False,
            ).strip()
            for s in sample_batch
        ]
        images = [[load_image(s["image"])] for s in sample_batch]
        encoded = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens == 0:
            raise ValueError("All labels were masked to -100. Check your dataset and processor.")
        print("✅ Processor masking test passed.")

    # ⛔ 학습 로직은 실행되지 않음
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(
        os.path.dirname(args.train_json),
        f"finetuned_medgemma_qlora_{timestamp}",
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=args.epochs,                                 # ✅ epochs 반영
            per_device_train_batch_size=args.batch_size,                  # ✅ CLI 인자 반영
            gradient_accumulation_steps=args.grad_accum,                  # ✅
            learning_rate=args.lr,                                        # ✅
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=True,
            logging_steps=args.logging_steps,                             # ✅
            save_strategy="steps",
            save_steps=args.save_steps,                                   # ✅
            eval_steps=args.eval_steps,                                   # ✅
            dataloader_num_workers=args.num_workers,                      # ✅
            ddp_find_unused_parameters=True,
            max_grad_norm=1.0,
            label_names=["labels"],
            report_to=["tensorboard"],
        ),
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=None,  # 콜백에서 직접 평가
        data_collator=lambda batch: collate_fn(batch, processor, ds_all),
        callbacks=[
            PrintLossCallback(),
            PrintEvalCallback(),
            EvalOnEpochEndCallback(ds_val, processor),
        ],
    )

    if accelerator.is_main_process:
        print("Starting training...")
    trainer.train()
  
    final_out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.train_json),
        f"medgemma_final_v.0.1.3",
        )   

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained(final_out_dir, safe_serialization=True)
        print(f"Adapter saved to {final_out_dir} as safetensors")

if __name__ == "__main__":
    main()


