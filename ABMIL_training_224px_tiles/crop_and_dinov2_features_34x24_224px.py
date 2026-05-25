#!/usr/bin/env python3
"""
Fast multi-GPU DINOv2 feature extraction - simplified for maximum speed.
"""

import os
os.environ["TQDM_NOTEBOOK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, AutoModel

# ==================
# CONFIG
# ==================
INPUT_DIR = Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/rig_v1_images/images_soil_geomanitoba")
OUTPUT_DIR = Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_features_7616x5440/dinov2_224_grid_34x24")
MODEL_LOCAL = "/home/thomas_plante_stcyr/workspace/torch/2_1_1/pretrained/dinov2"
MODEL_NAME = "facebook/dinov2-base"

CROP_LONG_PX = 288
CROP_SHORT_PX = 12
CROP_SIZE = 224
GRID_COLS = 34
GRID_ROWS = 24
BATCH_SIZE = 128      # Much larger batches

NUM_GPUS = 3
GPU_IDS = [0, 1, 2]

SEED_SALT = "jitter_v1"
SAVE_FLOAT16 = True
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

Image.MAX_IMAGE_PIXELS = None


# ==================
# HELPERS
# ==================
def compute_crop_box(w, h, crop_long, crop_short):
    long_is_w = w >= h
    if long_is_w:
        left, right = crop_long, w - crop_long
        top, bottom = crop_short, h - crop_short
    else:
        left, right = crop_short, w - crop_short
        top, bottom = crop_long, h - crop_long
    
    left = max(0, min(left, w // 2 - 1))
    right = max(left + 1, min(right, w))
    top = max(0, min(top, h // 2 - 1))
    bottom = max(top + 1, min(bottom, h))
    return (left, top, right, bottom)


def l2norm_torch(x, dim=-1):
    return torch.nn.functional.normalize(x, dim=dim)


def sha_seed(s: str) -> int:
    h = hashlib.sha1(s.encode()).hexdigest()[:8]
    return int(h, 16)


def global_resize_with_pad(img: Image.Image, size: int = CROP_SIZE) -> Image.Image:
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img2 = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img2, (ox, oy))
    return canvas


def grid_boxes_fixed(w, h, t=CROP_SIZE, cols=GRID_COLS, rows=GRID_ROWS, seed_key=""):
    if w < t or h < t:
        x0 = max(0, (w - t) // 2)
        y0 = max(0, (h - t) // 2)
        return [(x0, y0, x0 + t, y0 + t)], 0, 0, 0, 0

    sx = t if cols <= 1 else max(1, (w - t) // max(cols - 1, 1))
    sy = t if rows <= 1 else max(1, (h - t) // max(rows - 1, 1))

    rng = np.random.RandomState(sha_seed(seed_key) % (2**32 - 1))
    jx = 0 if cols <= 1 else int(rng.randint(0, sx))
    jy = 0 if rows <= 1 else int(rng.randint(0, sy))

    boxes = []
    for r in range(rows):
        for c in range(cols):
            x0 = min(w - t, jx + c * sx)
            y0 = min(h - t, jy + r * sy)
            boxes.append((x0, y0, x0 + t, y0 + t))
    return boxes, sx, sy, jx, jy


def get_num_register_tokens(m) -> int:
    return int(getattr(getattr(m, "config", object()), "num_register_tokens", 0))


def safe_save_npz_atomic(out_path, **arrays):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with NamedTemporaryFile(dir=out_path.parent, delete=False) as tmpf:
        tmp_name = tmpf.name
        np.savez_compressed(tmpf, **arrays)
        tmpf.flush()
        os.fsync(tmpf.fileno())
    
    os.replace(tmp_name, out_path)


def list_images(root: Path):
    return [p for p in sorted(Path(root).glob("**/*")) if p.suffix.lower() in EXTS]


# ==================
# EXTRACTION
# ==================
@torch.inference_mode()
def extract_features_simple(image_path: Path, device, model, processor, amp_ctx, num_regs):
    """Simple, fast extraction."""
    
    with Image.open(image_path) as _img:
        _img = ImageOps.exif_transpose(_img)
        orig_w, orig_h = _img.size
        
        # Crop image
        crop_box = compute_crop_box(orig_w, orig_h, CROP_LONG_PX, CROP_SHORT_PX)
        img = _img.crop(crop_box).convert("RGB")
        w, h = img.size
        
        # Global view
        gimg = global_resize_with_pad(img, CROP_SIZE)
        g_in = processor(images=gimg, return_tensors="pt", do_resize=False, do_center_crop=False)
        g_in["pixel_values"] = g_in["pixel_values"].to(device, dtype=model.dtype)
        
        with amp_ctx:
            g_out = model(**g_in).last_hidden_state
        
        g_cls = l2norm_torch(g_out[:, 0, :]).squeeze(0).cpu().float().numpy()
        del g_in, g_out
        
        # Grid tiles
        seed_key = f"{SEED_SALT}|{image_path.stem}|{w}x{h}|{GRID_COLS}x{GRID_ROWS}"
        boxes, sx, sy, jx, jy = grid_boxes_fixed(w, h, CROP_SIZE, GRID_COLS, GRID_ROWS, seed_key)
        
        cls_list, mean_list = [], []
        
        # Process in batches
        for i in range(0, len(boxes), BATCH_SIZE):
            batch_boxes = boxes[i:i + BATCH_SIZE]
            
            # Crop all tiles at once
            batch_imgs = [img.crop(b) for b in batch_boxes]
            
            # Process batch
            inputs = processor(images=batch_imgs, return_tensors="pt", do_resize=False, do_center_crop=False)
            inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=model.dtype)
            
            with amp_ctx:
                out = model(**inputs)
                last = out.last_hidden_state
            
            cls = l2norm_torch(last[:, 0, :])
            patches = last[:, 1 + num_regs:, :]
            mean_p = l2norm_torch(patches.mean(dim=1))
            
            cls_list.append(cls.cpu().float())
            mean_list.append(mean_p.cpu().float())
            
            del inputs, out, last, cls, patches, mean_p, batch_imgs
        
        cls_all = torch.cat(cls_list, dim=0).numpy()
        mean_all = torch.cat(mean_list, dim=0).numpy()
        coords = np.array(boxes, dtype=np.int32)
    
    if SAVE_FLOAT16:
        cls_all = cls_all.astype("float16")
        mean_all = mean_all.astype("float16")
        g_cls = g_cls.astype("float16")
    
    meta = {
        "original_size": f"{orig_w}x{orig_h}",
        "cropped_size": f"{w}x{h}",
        "crop_box": crop_box,
        "cols": GRID_COLS,
        "rows": GRID_ROWS,
        "stride_x": int(sx),
        "stride_y": int(sy),
        "jitter_x": int(jx),
        "jitter_y": int(jy),
        "model": MODEL_NAME,
        "crop_size": CROP_SIZE,
        "dtype": "float16" if SAVE_FLOAT16 else "float32",
        "patch_size": 14
    }
    
    return cls_all, mean_all, coords, g_cls, meta


def worker_process(gpu_id: int, image_paths: list, output_dir: Path, progress_queue):
    """Worker for one GPU."""
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    
    # Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except:
        pass
    
    # Load model
    processor = AutoImageProcessor.from_pretrained(MODEL_LOCAL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_LOCAL, local_files_only=True).to(device).eval()
    
    run_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.to(dtype=run_dtype)
    
    amp_ctx = torch.autocast(device_type="cuda", dtype=run_dtype, enabled=True)
    num_regs = get_num_register_tokens(model)
    
    # Process images
    for img_path in image_paths:
        out_filename = f"{img_path.stem}_dinov2_{CROP_SIZE}_grid_{GRID_COLS}x{GRID_ROWS}.npz"
        out_path = output_dir / out_filename
        
        if out_path.exists():
            progress_queue.put((gpu_id, True, None))
            continue
        
        try:
            cls, mean, coords, gcls, meta = extract_features_simple(
                img_path, device, model, processor, amp_ctx, num_regs
            )
            
            safe_save_npz_atomic(
                out_path,
                cls=cls,
                mean=mean,
                coords=coords,
                global_cls=gcls,
                meta=json.dumps(meta)
            )
            
            progress_queue.put((gpu_id, True, None))
            
        except Exception as e:
            progress_queue.put((gpu_id, False, str(e)))
            print(f"[GPU {gpu_id}] Failed on {img_path.name}: {e}")


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_images = list_images(INPUT_DIR)
    if not all_images:
        print("No images found.")
        return
    
    print(f"Found {len(all_images)} images")
    print(f"GPUs: {GPU_IDS}, Batch size: {BATCH_SIZE}")
    print(f"Grid: {GRID_COLS}x{GRID_ROWS} = {GRID_COLS*GRID_ROWS} tiles per image")
    
    # Split work
    images_per_gpu = [[] for _ in range(NUM_GPUS)]
    for i, img_path in enumerate(all_images):
        images_per_gpu[i % NUM_GPUS].append(img_path)
    
    for i, imgs in enumerate(images_per_gpu):
        print(f"GPU {GPU_IDS[i]}: {len(imgs)} images")
    
    progress_queue = mp.Queue()
    
    # Start workers
    processes = []
    for gpu_idx, gpu_id in enumerate(GPU_IDS):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, images_per_gpu[gpu_idx], OUTPUT_DIR, progress_queue)
        )
        p.start()
        processes.append(p)
    
    # Monitor
    start_time = time.time()
    completed = 0
    errors = 0
    last_print = time.time()
    
    while any(p.is_alive() for p in processes):
        try:
            gpu_id, success, error = progress_queue.get(timeout=0.5)
            if success:
                completed += 1
            else:
                errors += 1
            
            # Print every 5 seconds
            if time.time() - last_print >= 5:
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1e-6)
                eta = (len(all_images) - completed) / max(rate, 1e-6)
                print(f"{completed}/{len(all_images)} | {rate:.2f} img/s | ETA: {eta/60:.1f}min | Errors: {errors}")
                last_print = time.time()
        except:
            pass
    
    # Drain queue
    while not progress_queue.empty():
        gpu_id, success, error = progress_queue.get()
        if success:
            completed += 1
        else:
            errors += 1
    
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Done in {elapsed/60:.1f} min")
    print(f"Success: {completed}, Errors: {errors}")
    print(f"Average: {completed/max(elapsed, 1):.2f} img/s")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()