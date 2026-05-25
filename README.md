# abmil-optical-psd

Code and experiment scripts for predicting particle-size distributions (PSD) from optical images of soils and granular materials.

This repository supports two pipelines:

- **ABMIL** (Attention-Based Multiple Instance Learning), trained on frozen DINOv2 tile features
- **GraiNet baseline**, trained end-to-end on downscaled images

This repository **does not include** the dataset, precomputed features, or training outputs. Store those files **outside** the repository and pass paths to the scripts via command-line arguments.

---

## Repository structure

- `ABMIL_training_224px_tiles/`  
  ABMIL training and ablation scripts. These scripts operate on saved DINOv2 feature tensors.

- `Lang_grainet_training/`  
  GraiNet baseline preprocessing and training scripts. These scripts operate on downscaled images.

- `figures/`  
  Notebooks used to generate article figures.

- `requirements.txt`  
  Python dependencies used by the scripts.

---

## Setup

Python **3.10+** is recommended. Python **3.12** was tested.

### Create and activate a virtual environment

**Code cell (bash):**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

**Code cell (bash):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional sanity check

**Code cell (bash):**
```bash
python -c "import numpy, pandas, torch; print('OK'); print('CUDA available:', torch.cuda.is_available())"
```

---

## Workflow overview

To run the code in this repository:

1. Download the Photogranulometry dataset.
2. Keep the dataset **outside** the repository.
3. Choose one of the two pipelines below.

### ABMIL workflow

1. Extract frozen DINOv2 features from the dataset images.
2. Train ABMIL using the extracted features and PSD labels.

### GraiNet workflow

1. Downscale and preprocess the dataset images.
2. Build the label files required by the baseline.
3. Train the GraiNet baseline end-to-end.

---

## Dataset

This repository uses the **Photogranulometry** dataset:

Plante St-Cyr, T., Duhaime, F., Dubé, J., Grenier, S. (2025).  
*Photogranulometry - Dataset of soil images with corresponding particle size distributions.*  
Federated Research Data Repository.  
https://doi.org/10.20383/103.01316

Download and extract the dataset **outside** this repository.

### Required files

- Image files
- `PSD.xlsx`
- `MASS.xlsx` (only required for mass or moisture ablations)

### PSD.xlsx requirements

The PSD file must include:

- A sample ID column named `num_publication`
- PSD bin columns named with the format `%_Xmm`

---

## Recommended local layout

A typical local layout is:

```text
<DATA_ROOT>/
  images/
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

Notes:

- The `features/` folder is generated locally during ABMIL feature extraction.
- The `outputs/` folder is used for logs, checkpoints, predictions, and experiment results.

---

## ABMIL pipeline

The ABMIL pipeline uses a two-stage workflow:

- **Stage 1:** extract frozen DINOv2 features from image tiles
- **Stage 2:** train ABMIL on saved feature tensors and PSD labels

---

## ABMIL feature extraction

Inspect the available arguments:

**Code cell (bash):**
```bash
python ABMIL_training_224px_tiles/ABMIL_224px_tiles.py --help
```

After feature extraction, the feature root should contain a folder such as:

- `features/dinov2_224_grid_34x24/`

Resolution-ablation features may also be stored in folders such as:

- `features/Res075/`
- `features/Res050/`
- `features/Res025/`
- `features/Res010/`

---

## ABMIL training example

**Code cell (bash):**
```bash
python ABMIL_training_224px_tiles/ABMIL_224px_tiles.py \
  --plan moisture \
  --truth-xlsx /ABS/PATH/TO/labels/PSD.xlsx \
  --mass-xlsx /ABS/PATH/TO/labels/MASS.xlsx \
  --features-root /ABS/PATH/TO/features \
  --res-features-root /ABS/PATH/TO/features \
  --output-dir /ABS/PATH/TO/outputs
```

Outputs are written under the directory specified by `--output-dir`.

Use `--help` to inspect the available plans, path arguments, and defaults.

---

## DINOv2

ABMIL feature extraction uses DINOv2.

The checkpoint used for feature extraction was downloaded from:

https://huggingface.co/facebook/dinov2-base

If you use DINOv2 features, please cite:

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., and Bojanowski, P. (2024).  
*DINOv2: Learning Robust Visual Features without Supervision.*  
arXiv.  
https://doi.org/10.48550/arXiv.2304.07193

---

## GraiNet baseline

The GraiNet baseline is run in separate preprocessing, label-building, and training steps.

### Preprocessing

**Code cell (bash):**
```bash
python Lang_grainet_training/thomas_preprocess.py --help
```

### Label building

**Code cell (bash):**
```bash
python Lang_grainet_training/thomas_build_lang_labels.py --help
```

### Training

Training scripts are located in `Lang_grainet_training/`.

Use `--help` on the relevant training script to configure paths and options.

---

## Main entry points

- `ABMIL_training_224px_tiles/ABMIL_224px_tiles.py`
- `Lang_grainet_training/thomas_preprocess.py`
- `Lang_grainet_training/thomas_build_lang_labels.py`

Additional scripts may be present for ablations or exploratory analyses.

---

## Optional and exploratory scripts

- `ABMIL_training_224px_tiles/ABMIL_224px_tiles_with_feature_ablation.py`

This script contains feature-level and resolution-ablation utilities. It is kept for additional analyses and is not required for the main pipeline.

---

## Credits and third-party code

This repository includes a GraiNet baseline adapted from the original **GRAINet** code by Lang et al.

Upstream repository:

https://github.com/langnico/GRAINet

If you use or build on the GraiNet baseline, please cite:

Lang et al. (2021).  
*GRAINet: mapping grain size distributions in river beds from UAV images with convolutional neural networks.*  
Hydrology and Earth System Sciences.

Changes relative to the upstream code were kept minimal. Most modifications concern path handling, input/output formatting, and integration with the Photogranulometry dataset.

---

## What this repository does not include

This repository does not include:

- Dataset images
- PSD or mass spreadsheets
- Precomputed DINOv2 features
- Training outputs
- Model checkpoints

These files should be downloaded or generated locally and kept outside git.

---

## License

This project is licensed under the MIT License. See `LICENSE`.

---

## Citation

If you use the dataset, please cite:

```bibtex
@dataset{plante_st_cyr_photogranulometry_2025,
  author    = {Plante St-Cyr, Thomas and Duhaime, François and Dubé, Jean-Sébastien and Grenier, Simon},
  title     = {Photogranulometry - Dataset of soil images with corresponding particle size distributions},
  year      = {2025},
  publisher = {Federated Research Data Repository},
  doi       = {10.20383/103.01316},
  url       = {https://doi.org/10.20383/103.01316}
}
```

If you use the code or reproduce the experiments, please also cite the associated paper once it is available.

---

## Contact

For questions, issues, or citation information, please open an issue on GitHub.