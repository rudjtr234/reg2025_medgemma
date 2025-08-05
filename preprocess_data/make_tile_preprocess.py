# kb size preprocessing

import os, glob, shutil

SRC_TILE_DIR = "/your_path_directory/REG_2025_tile_preprocess_final_v.0.2.0"
DEST_TILE_DIR = "/your_path_directory/REG_2025_tile_preprocess_final_v.0.2.1"
MIN_SIZE_KB = 300

os.makedirs(DEST_TILE_DIR, exist_ok=True)

slide_dirs = sorted([
    d for d in os.listdir(SRC_TILE_DIR)
    if os.path.isdir(os.path.join(SRC_TILE_DIR, d))
])

print(f"📊 전체 슬라이드 디렉토리 수: {len(slide_dirs)}")

processed_slides = []
skipped_slides = []

for idx, slide_dir in enumerate(slide_dirs, 1):
    dest_slide_path = os.path.join(DEST_TILE_DIR, slide_dir)

    if os.path.exists(dest_slide_path):
        print(f"[{idx}/{len(slide_dirs)}] ⏩ SKIP: {slide_dir} → 이미 존재하여 건너뜀")
        skipped_slides.append(slide_dir)
        continue

    os.makedirs(dest_slide_path, exist_ok=True)
    src_slide_path = os.path.join(SRC_TILE_DIR, slide_dir)
    tile_paths = glob.glob(os.path.join(src_slide_path, "*.jpg"))
    kept_count = 0

    for tile_path in tile_paths:
        if os.path.getsize(tile_path) > MIN_SIZE_KB * 1024:
            shutil.copy(tile_path, os.path.join(dest_slide_path, os.path.basename(tile_path)))
            kept_count += 1

    if kept_count == 0:
        print(f"[{idx}/{len(slide_dirs)}] ⚠️  WARN: {slide_dir} → 복사된 타일 없음")
    else:
        print(f"[{idx}/{len(slide_dirs)}] ✓ OK:   {slide_dir} → {kept_count}개 타일 복사됨")

    processed_slides.append(slide_dir)

print("\n🎯 요약")
print(f"✅ 처리된 디렉토리: {len(processed_slides)}개")
print(f"⏩ 스킵된 디렉토리: {len(skipped_slides)}개")
print(f"→ 결과 경로: {DEST_TILE_DIR}")

# 스킵된 디렉토리 목록 출력 (옵션)
if skipped_slides:
    print("\n📄 스킵된 디렉토리 목록:")
    for d in skipped_slides:
        print(f"  - {d}")



