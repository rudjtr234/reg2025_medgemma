import json
from eval import REG_Evaluator

GT_PATH = '/home/mts/ssd_16tb/member/jks/reg2025_medgemma/evaluation/reg/metric/json/ground_truth_all.json'
PRED_PATH = '/home/mts/ssd_16tb/member/jks/reg2025_medgemma/evaluation/reg/metric/json/clean_report_v0.1.0.json'

EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'
SPACY_MODEL = 'en_core_sci_lg'

evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)

# 데이터 불러오기
with open(GT_PATH, 'r') as f:
    gt_data = json.load(f)

with open(PRED_PATH, 'r') as f:
    pred_data = json.load(f)

# 예측 결과를 딕셔너리로 변환
pred_dict = {item['id']: item['report'] for item in pred_data}

# 평가 쌍 구성 (예측이 존재하는 경우에만)
eval_pairs = []
missing = 0

for item in gt_data:
    gt_id = item['id']
    gt_report = item['report']

    if gt_id not in pred_dict:
        missing += 1
        continue  # 예측이 없는 항목은 건너뜀

    pred_report = pred_dict[gt_id]
    eval_pairs.append((gt_report, pred_report))

# 평가 수행
score = evaluator.evaluate_dummy(eval_pairs)

print(f"\n평가 쌍 개수: {len(eval_pairs):,}")
print(f"예측 누락 ID 수: {missing:,}")
print(f"\n평균 Ranking Score: {score:.4f}")
