#!/usr/bin/env python3

"""
Label Transformation Pipeline and Mathematical Rationale

This script extracts raw sieve analysis data and converts it into standardized Cumulative 
Distribution Functions (CDFs) and Mean Diameters (dm) suitable for ground truth labels. 

Transformation Steps:
1. Extraction & Sorting: 
   Sieve sizes and passing percentages are extracted from the source data and sorted 
   in ascending order to ensure monotonic progression.
   
2. Scale Normalization:
   Passing percentages are divided by 100.0 to map the domain from [0, 100] to [0.0, 1.0], 
   satisfying the mathematical requirements of a CDF.

3. Log-Space Interpolation (Optimized Range):
   The physical sample data is mapped onto a 22-element grid (`LANG_EDGES_MM`) spanning 
   0.05 mm to 80 mm. These edges are equally spaced in logarithmic space (log-x). 
   Interpolation is performed in log-space (`np.log`) to maintain geometric accuracy 
   across the distribution.

4. Monotonic Enforcement:
   `np.maximum.accumulate` and `np.clip` act as computational safeguards to enforce 
   strict monotonicity and bounds [0.0, 1.0], eliminating floating-point interpolation artifacts.

5. Geometric Mean for Representative Bin Diameters:
   When calculating the Mean Diameter (dm), the PDF is derived by differentiating the CDF. 
   Since the bins are strictly log-spaced, the representative diameter of each bin ($d_{mi}$) 
   is calculated using the geometric mean ($\sqrt{d_{lower} \times d_{upper}}$). This is 
   the mathematically correct descriptor for log-normal particle distributions.

6. Direct Expected Value Calculation for dm:
   The calculated PDF represents a mass-weighted distribution. The scalar Mean Diameter 
   (dm) is calculated directly as the expected value ($\sum (p_i \times d_{mi})$). 
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import re

ROOT = Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch")
IMG_DIR = ROOT / "baseline_other_papers" / "Lang_grainet" / "thomas_images_500x200"
PSD_XLSX = ROOT / "scratch" / "labels" / "PSD.xlsx"
OUT_NPZ = ROOT / "baseline_other_papers" / "Lang_grainet" / "data_global_from_PSD_500x200.npz"
SAMPLE_ID_COLUMN = "num_publication"
SID_RX = re.compile(r"^(?P<sid>\d{1,6})")
TARGET_ROWS = 500
TARGET_COLS = 200

# 22 edges create 21 equally spaced bins in log(x) from 0.05 to 80 mm
LANG_EDGES_MM = np.logspace(np.log10(0.05), np.log10(80), 22)

def build_lang_cdf_from_row(row: pd.Series) -> np.ndarray:
    sieve_sizes_mm = []
    percents = []

    for col, val in row.items():
        if isinstance(col, str) and col.startswith("%_") and col.endswith("mm"):
            size_str = col[2:-2]
            try:
                size_mm = float(size_str)
            except ValueError:
                continue

            if pd.isna(val):
                continue

            sieve_sizes_mm.append(size_mm)
            percents.append(float(val))

    if not sieve_sizes_mm:
        raise ValueError(f"No %_Xmm columns found in row, row index={row.name}")

    sieve_sizes_mm = np.array(sieve_sizes_mm, dtype=float)
    percents = np.array(percents, dtype=float)

    order = np.argsort(sieve_sizes_mm)
    sieve_sizes_mm = sieve_sizes_mm[order]
    percents = percents[order]

    cdf_values = percents / 100.0

    # Ensure sieve sizes are positive for log-interpolation
    nonzero_mask = sieve_sizes_mm > 0
    s_mm_log_safe = sieve_sizes_mm[nonzero_mask]
    c_val_log_safe = cdf_values[nonzero_mask]

    if len(s_mm_log_safe) == 0:
        raise ValueError("No non-zero sieve sizes available for log interpolation.")

    # Interpolate directly on the log-spaced edges
    cdf_interp = np.interp(
        np.log(LANG_EDGES_MM),
        np.log(s_mm_log_safe),
        c_val_log_safe,
        left=c_val_log_safe[0],
        right=c_val_log_safe[-1]
    )

    cdf_interp = np.maximum.accumulate(cdf_interp)
    cdf_interp = np.clip(cdf_interp, 0.0, 1.0)
    return cdf_interp


def dm_from_cdf_lang(cdf_lang: np.ndarray) -> float:
    # Convert mm to m for standard SI calculation
    edges_m = LANG_EDGES_MM / 1000.0
    pdf = np.diff(cdf_lang) 

    d_lower = edges_m[:-1]
    d_upper = edges_m[1:]
    
    # Geometric mean is valid for all bins as lower bound > 0
    dmi = np.sqrt(d_lower * d_upper)

    if pdf.sum() <= 0:
        return float(dmi.mean() * 100.0)

    # Normalized PDF expected value
    pdf_normalized = pdf / pdf.sum()
    dm_m = np.sum(pdf_normalized * dmi)
    
    # Return dm in cm (standard for this specific baseline reporting)
    return float(dm_m * 100.0)


def load_psd_xlsx(psd_path: Path):
    df = pd.read_excel(psd_path)
    if SAMPLE_ID_COLUMN not in df.columns:
        raise ValueError(f"Column {SAMPLE_ID_COLUMN!r} not found in {psd_path}")

    df = df.dropna(subset=[SAMPLE_ID_COLUMN])

    psd_cdf_map = {}
    dm_map = {}

    for _, row in df.iterrows():
        try:
            sid = int(row[SAMPLE_ID_COLUMN])
            cdf_lang = build_lang_cdf_from_row(row)
            dm_cm = dm_from_cdf_lang(cdf_lang)
            psd_cdf_map[sid] = cdf_lang
            dm_map[sid] = dm_cm
        except ValueError:
            continue

    return psd_cdf_map, dm_map


def extract_sid_from_name(name: str) -> int | None:
    m = SID_RX.match(name)
    if not m:
        return None
    return int(m.group("sid"))


def main():
    print("Loading PSD.xlsx...")
    psd_cdf_map, dm_map = load_psd_xlsx(PSD_XLSX)
    print(f"Loaded PSD for {len(psd_cdf_map)} samples from {PSD_XLSX}")

    print("Indexing images in:", IMG_DIR)
    img_paths = sorted(
        p for p in IMG_DIR.glob("**/*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )

    images = []
    histograms = []
    dms = []
    names = []

    missing_psd = 0
    bad_shape = 0
    total = 0

    for img_path in img_paths:
        total += 1
        fname = img_path.name
        sid = extract_sid_from_name(fname)
        if sid is None or sid not in psd_cdf_map:
            missing_psd += 1
            continue

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        if arr.shape[0] != TARGET_ROWS or arr.shape[1] != TARGET_COLS or arr.shape[2] != 3:
            bad_shape += 1
            continue

        images.append(arr)
        histograms.append(psd_cdf_map[sid])
        dms.append(dm_map[sid])
        names.append(img_path.stem)

    if not images:
        raise RuntimeError("No images matched PSD.xlsx with proper shape.")

    images = np.stack(images, axis=0)
    histograms = np.stack(histograms, axis=0)
    dms = np.array(dms, dtype=np.float32)
    names = np.array(names)

    print(f"Total images found: {total}")
    print(f"Used in dataset:   {len(images)}")
    print(f"Missing PSD:       {missing_psd}")
    print(f"Bad shape:         {bad_shape}")
    print("Final shapes:")
    print("  images     :", images.shape)
    print("  histograms :", histograms.shape)
    print("  dm         :", dms.shape)
    print("  tile_names :", names.shape)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        images=images,
        histograms=histograms,
        dm=dms,
        tile_names=names,
    )
    print("Saved:", OUT_NPZ)


if __name__ == "__main__":
    main()