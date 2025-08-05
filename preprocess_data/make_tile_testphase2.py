import os, glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile
import openslide


ImageFile.LOAD_TRUNCATED_IMAGES = True


# .tiff --> tile


# ── 설정 ────────────────────────────────────────────
WSI_DIR   = "/data/dataset/REG_2025_dataset/REG_2025_Testphase2_08_04/REG_test2_v0.1.0"
TILE_ROOT = "/data/dataset/REG_2025_dataset/REG_2025_06_09_final/preprocess_tile/testphase2_dataset_v.0.1.0"

TILE_PX   = 896
STRIDE    = TILE_PX
LEVEL     = 0

os.makedirs(TILE_ROOT, exist_ok=True)

# ── 타일 생성 ───────────────────────────────────────
def generate_tiles(wsi_path):
    slide_name = wsi_path.stem
    out_dir = Path(TILE_ROOT) / slide_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except openslide.OpenSlideError as e:
        print(f"[{slide_name}] ❌ 슬라이드 열기 실패: {e}")
        return

    w, h = slide.level_dimensions[LEVEL]
    tile_count = 0

    for y in tqdm(range(0, h, STRIDE), desc=f"[{slide_name}] rows"):
        for x in range(0, w, STRIDE):
            tile = slide.read_region((x, y), LEVEL, (TILE_PX, TILE_PX)).convert("RGB")
            tile_path = out_dir / f"{slide_name}_{x}_{y}.jpg"
            tile.save(tile_path, format="JPEG", quality=95)
            tile_count += 1

    print(f"[{slide_name}] ✅ 타일 저장 완료: {tile_count}개")

# ── 전체 슬라이드 반복 ───────────────────────────────
for wsi_path in sorted(Path(WSI_DIR).glob("*.tiff")):
    generate_tiles(wsi_path)

print("✅ 전체 슬라이드 타일링 완료")
