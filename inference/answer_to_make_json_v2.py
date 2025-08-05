#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is a provided example of how to generate pathology reports
# for whole-slide images using the MedGemma model. It demonstrates how
# to load the merged model and run inference across multiple GPUs. The
# script includes functions for merging model adapters, generating
# partial reports in parallel, and aggregating results into final
# representative reports per slide. Note that file paths and model
# identifiers are environment-specific and may need to be adjusted for
# different setups.

import os
import glob
import json
import gc
import re
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel


import re

# 안전한 re.sub 래퍼 – 패턴 오류가 나도 원본을 그대로 돌려줌
def safe_sub(pattern, repl, text, flags=0):
    try:
        return re.sub(pattern, repl, text, flags=flags)
    except re.error:
        return text

def clean_report(text: str) -> str:
    """
    모델이 생성한 리포트를 후처리:
    ① HTML 태그 제거 ② 개행 정규화 ③ 'n/a' 제거
    ④ 중복 라인 제거 ⑤ 'No tumor present.' 1회만 유지
    """
    # 1) HTML 태그 제거
    text = safe_sub(r'<[^>]+>', '', text)

    # 2) 개행 정규화
    text = text.replace('\\n\\n', '\n\n').replace('\\n', '\n')

    # 3) 'n/a' 변형 제거
    text = safe_sub(r'\b[nN]/?[aA]\b', '', text, flags=re.IGNORECASE)

    # 4) 중복 라인 제거 + 프롬프트 잔여 헤더 제거
    pattern_hdr = re.compile(
        r'^#*\s*(Human|Assistant|Response|Report|Example)\s*[:：]?\s*',
        flags=re.IGNORECASE
    )

    seen, new_lines = set(), []
    for line in text.split('\n'):
        line = pattern_hdr.sub('', line.strip())
        if line and line not in seen:
            seen.add(line)
            new_lines.append(line)

    # 5) 'No tumor present.' 한 번만 남기기
    filtered = []
    for l in new_lines:
        if l == "No tumor present." and "No tumor present." in filtered:
            continue
        filtered.append(l)

    return '\n'.join(filtered).strip()


# Base paths for model and data (example values; adjust to your environment)

from pathlib import Path

BASE_ID    = Path("/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/medgemma-4b-it")
ADAPTER_ID = Path("/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/make_json/train_json/medgemma_final_v.0.1.0")
MERGE_DIR  = Path("/home/mts/ssd_16tb/member/jks/medgemma_reg2025/models/medgemma-merged_final_v0.1.0")
IMG_DIR    = Path("/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/REG_2025_tile_preprocess_final_v.0.2.0/")
OUT_JSON   = Path("/home/mts/ssd_16tb/member/jks/medgemma_reg2025/notebooks/data/preprocess_tile/fintuning_result/medgemma_final_v0.1.0.json")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Environment variables to avoid certain NCCL issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# System prompt for the model, instructing it to generate structured pathology reports
SYSTEM_PROMPT = (
    "You are a pathologist AI assistant trained to analyze whole slide images (WSI).\n"
    "You are able to understand the visual content of histopathological slides and generate structured pathology reports.\n"
    "Your response must follow this exact format:\n"
    "[Organ/tissue], [procedure];\n"
    "[Findings]\n"
    "- If there are multiple findings, number them (1., 2., etc.), each on a new line.\n"
    "- Add notes only after a double newline (\\n\\n), and only if directly relevant to the specimen (e.g., \"Note) The specimen includes muscle proper.\")\n"
    "- For non-malignant cases, state: \"No tumor present.\"\n"
    "Strictly output only the report. Do not provide any explanation, commentary, or analysis beyond the formatted report itself. "
    "Do not refer to the image, model, reasoning, or any additional information.\n"
    "Examples (do not describe them, just use structure):\n"
    "Prostate, biopsy;\nAcinar adenocarcinoma, Gleason score 7 (4+3), grade group 3\n"
    "Breast, biopsy;\nMucinous carcinoma\n"
    "Urinary bladder, transurethral resection;\n1. Non-invasive papillary urothelial carcinoma, high grade\n2. Urothelial carcinoma in situ\n\n"
    "Note) The specimen includes muscle proper.\n"
    "Do not copy the above examples. Generate a report only based on the current image."
    "Output must follow the exact format above without any additional text or remarks."
)

# Generation parameters for the model
GEN_KWARGS = dict(
    max_new_tokens=128,
    num_beams=10,
    do_sample=False,
    repetition_penalty=1.2
)


def merge_adapter_once():
    """
    Merge the adapter weights into the base model and save the merged model
    into MERGE_DIR. This function checks whether the merged model already
    exists before attempting to merge. It includes creation of a new
    language modeling head to ensure proper alignment of embedding weights.
    """
    model_patterns = ["pytorch_model.bin", "model.safetensors", "pytorch_model-*.bin"]

    def model_exists() -> bool:
        for pattern in model_patterns:
            if "*" in pattern:
                if any(Path(MERGE_DIR).glob(pattern)):
                    return True
            else:
                if (Path(MERGE_DIR) / pattern).exists():
                    return True
        return False

    if Path(MERGE_DIR).exists():
        if model_exists():
            print(f"Merged model already exists in {MERGE_DIR}")
            return
        else:
            print("Directory exists but merged model is not found. Proceeding to merge.")
    else:
        print("Starting adapter merge process...")

    try:
        # Load base model
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            tie_word_embeddings=False,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Load and merge PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        model = model.merge_and_unload()

        # Create a new lm_head using the embedding weights
        import torch.nn as nn
        new_lm_head = nn.Linear(
            model.language_model.embed_tokens.weight.shape[0],
            model.language_model.embed_tokens.weight.shape[1],
            bias=False
        ).to(dtype=torch.bfloat16)
        new_lm_head.weight.data = model.language_model.embed_tokens.weight.data.clone().detach()
        model.lm_head = new_lm_head

        # Save the merged model
        Path(MERGE_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MERGE_DIR, safe_serialization=True)
        processor = AutoProcessor.from_pretrained(BASE_ID, local_files_only=True)
        processor.save_pretrained(MERGE_DIR)

    except Exception as e:
        print(f"Error occurred during adapter merging: {e}")
        return

    # Verify the merged model
    if model_exists():
        print(f"Adapter merge completed successfully: {MERGE_DIR}")
    else:
        print("Merge appears incomplete. Please verify the contents of the merge directory.")


def generate_partial_reports(gpu_id: int, image_paths: list) -> None:
    """
    Generate pathology reports for a subset of images on a specific GPU.
    This function loads the model on the given GPU, processes the images,
    generates reports, and stores them in a temporary JSON file.

    Args:
        gpu_id (int): Identifier for the GPU to use.
        image_paths (list): List of image file paths to process.
    """
    print(f"[GPU {gpu_id}] Starting inference on {len(image_paths)} images")
    device = f"cuda:{gpu_id}"

    model = AutoModelForImageTextToText.from_pretrained(
        MERGE_DIR, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    processor = AutoProcessor.from_pretrained(MERGE_DIR)

    results = []
    total = len(image_paths)
    prev_time = time.time()

    for i, path in enumerate(image_paths, start=1):
        try:
            image = Image.open(path).convert("RGB")
            print(f"[GPU {gpu_id}] Loaded image: {path}", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to load image: {path} - {e}", flush=True)
            continue

        try:
            prompt = SYSTEM_PROMPT + "\n<start_of_image>"
            image_inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            if "input_ids" not in image_inputs or "pixel_values" not in image_inputs:
                print(f"[GPU {gpu_id}] Missing input_ids or pixel_values: {path}", flush=True)
                continue
            input_ids = image_inputs["input_ids"].to(device)
            pixel_values = image_inputs["pixel_values"].to(device)
            print(f"[GPU {gpu_id}] input_ids shape: {input_ids.shape}, pixel_values shape: {pixel_values.shape}", flush=True)
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to prepare inputs: {path} - {e}", flush=True)
            continue

        try:
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    **GEN_KWARGS
                )
                output_ids = gen_out.sequences[0]
            input_len = input_ids.shape[1]
            gen_output = output_ids[input_len:]
            response = processor.tokenizer.decode(gen_output, skip_special_tokens=True)
            cleaned = clean_report(response)
            results.append({
                "id": os.path.splitext(os.path.basename(path))[0] + ".tiff",
                "report": cleaned
            })
        except Exception as e:
            print(f"[GPU {gpu_id}] Error during generation or decoding: {path}: {e}", flush=True)
            continue

        if i % 10 == 0 or i == total:
            now = time.time()
            print(f"[GPU {gpu_id}] Completed {i}/{total} images ({(i/total)*100:.1f}%) - Last 10 images took {now - prev_time:.1f}s")
            prev_time = now

    tmp_file = f"{OUT_JSON}.gpu{gpu_id}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[GPU {gpu_id}] Finished. Temporary results saved to {tmp_file}")


import random
from collections import defaultdict, Counter
import multiprocessing as mp
import time, os, glob, json
from pathlib import Path

def generate_reports_multi_gpu(num_gpus: int = 4, *, num_dirs: int = 100, seed: int = 42) -> None:
    """
    Randomly sample `num_dirs` slide folders under IMG_DIR, gather all JPG tiles
    inside them, and distribute the work across multiple GPUs.

    Args:
        num_gpus (int):  how many GPUs to use.
        num_dirs (int):  how many slide directories to sample randomly.
        seed (int):      random seed for reproducibility.
    """
    # 1) 모든 슬라이드 디렉터리 수집 후 무작위 추출
    all_slide_dirs = glob.glob(os.path.join(IMG_DIR, "*"))
    random.seed(seed)
    slide_dirs = random.sample(all_slide_dirs, k=min(num_dirs, len(all_slide_dirs)))

    # 2) 선택된 디렉터리 안의 JPG 파일 모으기
    jpg_paths = []
    for slide_dir in slide_dirs:
        jpg_paths.extend(sorted(glob.glob(os.path.join(slide_dir, "*.jpg"))))

    print(f"\n📊 총 {len(jpg_paths):,}장의 이미지를 {num_gpus}개의 GPU로 나누어 추론합니다.\n")

    # 3) 경로를 GPU 수만큼 균등 분할
    chunk_size = len(jpg_paths) // num_gpus
    chunks = [jpg_paths[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    chunks.append(jpg_paths[(num_gpus - 1) * chunk_size:])

    processes, start_time = [], time.time()

    # 4) 멀티-프로세스 실행
    for gpu_id, paths in enumerate(chunks):
        print(f"🚀 GPU {gpu_id} → {len(paths):,}장 할당됨")
        p = mp.Process(target=generate_partial_reports, args=(gpu_id, paths))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 5) 임시 결과 병합
    final_results = []
    for gpu_id in range(num_gpus):
        tmp_file = f"{OUT_JSON}.gpu{gpu_id}.tmp"
        print(f"📥 GPU {gpu_id} 결과 병합 중: {tmp_file}")
        with open(tmp_file, "r", encoding="utf-8") as f:
            final_results.extend(json.load(f))
        os.remove(tmp_file)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 타일 단위 리포트 {len(final_results):,}개 저장 완료 → {OUT_JSON}")

    # 6) 슬라이드 대표 리포트 산출
    slide_report_map = defaultdict(list)
    for r in final_results:
        tile_id  = os.path.basename(r["id"])
        slide_id = "_".join(tile_id.split("_")[:-2]) + ".tiff"
        slide_report_map[slide_id].append(r["report"])

    representative_results = [
        {"id": sid, "report": Counter(reps).most_common(1)[0][0]}
        for sid, reps in slide_report_map.items()
    ]

    rep_out_json = OUT_JSON.replace(".json", "_representative.json")
    with open(rep_out_json, "w", encoding="utf-8") as f:
        json.dump(representative_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"📁 슬라이드 대표 리포트 {len(representative_results):,}개 저장 완료 → {rep_out_json}")
    print(f"⏱ 전체 소요 시간: {elapsed/60:.2f}분 ({elapsed:.1f}초)\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    merge_adapter_once()
    generate_reports_multi_gpu(num_gpus=4)
