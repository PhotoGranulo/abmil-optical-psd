# abmil-optical-psd

Code and experiment scripts for predicting particle-size distributions (PSD) from optical images using:
- **ABMIL** on DINOv2 tile features
- **GraiNet baseline** pipelines for comparison

## Repository purpose
This repository contains training, preprocessing, and ablation scripts used to study optical PSD prediction from labeled soil/granular image datasets.

## Repository structure
- `ABMIL_training_224px_tiles/`: ABMIL training and ablation scripts
- `Lang_grainet_training/`: GraiNet baseline preprocessing and training scripts
- `figures/`: notebooks for article figures

## Setup
Use Python 3.10+ (3.12 tested in this environment).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data and provenance (required inputs)
All data paths are now configurable via CLI arguments or environment variables.

### Required files
1. `PSD.xlsx` (ground-truth PSD labels)
   - Required columns include sample ID (`num_publication`) and `%_Xmm` PSD columns.
2. `MASS.xlsx` (sample mass metadata)
   - Required for moisture/mass-feature ablations.
3. Image folders (raw or resized)
   - Raw images for feature extraction/preprocessing.
4. Feature folders (DINOv2 feature tensors)
   - `dinov2_224_grid_34x24/`
   - resolution ablation folders: `Res075/`, `Res050/`, `Res025/`, `Res010/`

### Suggested local layout
```text
<DATA_ROOT>/
  labels/
    PSD.xlsx
    MASS.xlsx
  features/
    dinov2_224_grid_34x24/
    Res075/
    Res050/
    Res025/
    Res010/
  outputs/
```

### Acquisition/access notes
- `PSD.xlsx` and `MASS.xlsx` are project-internal label spreadsheets and are not versioned in this repo.
- Image datasets and extracted feature tensors are external large artifacts; keep them outside git and provide paths at runtime.
- If feature extraction is needed, use scripts in `ABMIL_training_224px_tiles/` and provide model/data locations explicitly.

## Reproduce one core result (ABMIL moisture ablation)
Example command:

```bash
python /home/runner/work/abmil-optical-psd/abmil-optical-psd/ABMIL_training_224px_tiles/ABMIL_224px_tiles.py \
  --plan moisture \
  --truth-xlsx /ABS/PATH/TO/labels/PSD.xlsx \
  --mass-xlsx /ABS/PATH/TO/labels/MASS.xlsx \
  --features-root /ABS/PATH/TO/features \
  --res-features-root /ABS/PATH/TO/features \
  --output-dir /ABS/PATH/TO/outputs
```

Output run folders are written under `--output-dir`.

## Main script path configuration
The following entry scripts support configurable input/output paths via CLI and/or environment:
- `ABMIL_training_224px_tiles/ABMIL_224px_tiles.py`
- `ABMIL_training_224px_tiles/ABMIL_224px_tiles_with_feature_ablation.py`
- `Lang_grainet_training/thomas_preprocess.py`
- `Lang_grainet_training/thomas_build_lang_labels.py`

Use `--help` on each script to view available options.
