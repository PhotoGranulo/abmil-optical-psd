#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps
import multiprocessing as mp
from functools import partial

# ====== CONFIG ======
IN_DIR = Path("torch/2_1_1/scratch/rig_v1_images/images_soil_geomanitoba")
OUT_DIR = Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/grainet_images_500x200")

TARGET_ROWS = 500  # height
TARGET_COLS = 200  # width

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Number of parallel workers; None => use mp.cpu_count()
NUM_WORKERS = None
# ====================


def process_one_image(in_path: Path, in_root: Path, out_root: Path):
    """
    Process a single image: load, exif transpose, resize to 200x500, save.
    Keeps relative subdirectory structure from IN_DIR under OUT_DIR.
    """
    try:
        rel_path = in_path.relative_to(in_root)
        out_path = out_root / rel_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(in_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        # resize(width, height)
        img_resized = img.resize((TARGET_COLS, TARGET_ROWS), Image.BICUBIC)
        img_resized.save(out_path)

        return True, str(in_path), None
    except Exception as e:
        return False, str(in_path), str(e)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(IN_DIR.glob("**/*")) if p.suffix.lower() in EXTS]

    if not files:
        print(f"No images found in {IN_DIR}")
        return

    print(f"Found {len(files)} images in {IN_DIR}")
    workers = NUM_WORKERS or mp.cpu_count()
    print(f"Using {workers} workers")

    worker_fn = partial(process_one_image, in_root=IN_DIR, out_root=OUT_DIR)

    success = 0
    fail = 0

    # chunksize helps reduce overhead for large lists
    chunksize = max(1, len(files) // (workers * 4))

    with mp.Pool(processes=workers) as pool:
        for i, (ok, img_path, err) in enumerate(pool.imap_unordered(worker_fn, files, chunksize=chunksize), start=1):
            if ok:
                success += 1
            else:
                fail += 1
                print(f"[ERROR] {img_path}: {err}")

            if i % 100 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] done | success: {success}, errors: {fail}")

    print("\nDone.")
    print(f"Total success: {success}, errors: {fail}")
    print("Resized images saved to:", OUT_DIR)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()