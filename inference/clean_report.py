import json
from collections import defaultdict, Counter

def extract_dirname(file_id):
    # 예: PIT_01_05661_01_1792_0.tiff → PIT_01_05661_01
    parts = file_id.replace(".tiff", "").split("_")
    return "_".join(parts[:4])  # 디렉토리 단위까지만 유지

def process_reports_by_directory(data):
    """
    디렉토리별로 가장 많이 등장한 리포트를 대표로 뽑기
    """
    dir_to_reports = defaultdict(list)

    # 각 디렉토리 단위로 리포트 수집
    for item in data:
        dirname = extract_dirname(item["id"])
        dir_to_reports[dirname].append(item["report"].strip())

    # 대표 리포트 생성
    result = []
    for dirname, reports in dir_to_reports.items():
        most_common = Counter(reports).most_common(1)[0][0]
        result.append({
            "id": f"{dirname}.tiff",  # 디렉토리 대표 이름으로 확장자 포함
            "report": most_common
        })

    return result

# 실행
with open("medgemma_final_v0.1.0.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed = process_reports_by_directory(raw_data)

with open("medgemma_final_v0.1.1.json", "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"✅ 디렉토리 수: {len(processed)}개 → 대표 리포트 저장 완료")
