from __future__ import annotations
import os
import re
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Sequence, Callable, Collection, Mapping
import math
import argparse
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", message=".*epoch parameter in `scheduler.step.*")

# python ABMIL_224px_tiles.py --plan moisture

# ==============================================================================
#                 User Adjustments - Paths and Run Configuration                            
# ==============================================================================

IN_CONTAINER = Path("/workspace").exists()
if IN_CONTAINER:
    ROOT = Path("/workspace")
else:
    ROOT = Path.home() / "workspace" / "torch" / "2_1_1"

TRUTH_XLSX = os.environ.get("PSD_TRUTH_XLSX", str(ROOT / "scratch" / "labels" / "PSD.xlsx"))
MASS_XLSX = os.environ.get("PSD_SAMPLE_MASS_XLSX",str(ROOT / "scratch" / "labels" / "MASS.xlsx"))
FEAT_ROOT = ROOT / "scratch" / "dinov2_features_7616x5440"
OUT_BASE  = ROOT / "model_runs_v2_ABMIL_224px_tiles"

MAX_MASS_G = 9000.0  # Used to normalize sample mass features

CFG = {
    "K_FOLDS": 5,               
    "SEED": 1337,               
    "EPOCHS": 100,              
    "PATIENCE": 10,             
    "BATCH_SIZE": 1,            
    "BAG_K": 4,                 
    "LR": 1e-3,                 
    "WEIGHT_DECAY": 1e-4,       
    "LOSS": "cvm",              
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "USE_TB": True,
    "USE_AMP": torch.cuda.is_available(),
    "POS_FOURIER": 8,           
    "ADD_GLOBAL_TILE": True,    
}

REPEATS = 3                     
IMAGES_PER_TRAY_SWEEP: Tuple[int, ...] = (1, 3, 6) 

# ==============================================================================
#                      User Adjustments - Naming Conventions
# ==============================================================================

SID_RX = re.compile(r"^(?P<sid>\d{1,6})")                   
TRAY_RX = re.compile(r"(?:^|_)tray(\d{1,4})(?:_|$)", re.I)  
MOIST_RX = re.compile(r"(?:^|_)(moist|wet)(?:_|$)", re.I)   
DRY_RX = re.compile(r"(?:^|_)dry(?:_|$)", re.I)             

SAMPLE_ID_COLUMN = r"num_publication"
PSD_COLUMNS = {
    "%_80mm":   80.0, "%_56mm":   56.0, "%_40mm":   40.0, "%_31.5mm": 31.5,
    "%_28mm":   28.0, "%_20mm":   20.0, "%_14mm":   14.0, "%_10mm":   10.0,
    "%_5mm":     5.0, "%_2.5mm":   2.5, "%_2mm":     2.0, "%_1.25mm":  1.25,
    "%_0.63mm":  0.63, "%_0.315mm": 0.315, "%_0.16mm":  0.16, "%_0.08mm":  0.08, "%_0.05mm":  0.05,
}
MOIST_MASS_COLUMN = r"moist.*mass"
DRY_MASS_COLUMN   = r"dry.*mass"

def _parse_mm(col) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm\b", str(col), re.I)
    return float(m.group(1)) if m else None

def _to_float_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^\d\.\-eE+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

MONTREAL_OFFSET = timedelta(hours=5)

# ==============================================================================
#                                Tile-Size Setups                            
# ==============================================================================

@dataclass(frozen=True)
class RowSetup:
    row_id: str
    feat_roots: Tuple[Path, ...]
    tiles_hw: Tuple[int, int]
    px: int

ROW_SETUPS: Dict[str, RowSetup] = {
    # For the following ablations: sample count, img per bag, moisture
    "T224": RowSetup("T224", (Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_features_7616x5440/dinov2_224_grid_34x24"),), (24, 34), 224),

    # Resolution setups for the ablation (100% already above)
    "T224_75":  RowSetup("T224_75",  (Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_224_res75_50_25/Res075"),), (18, 25), 224),
    "T224_50":  RowSetup("T224_50",  (Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_224_res75_50_25/Res050"),), (12, 17), 224),
    "T224_25":  RowSetup("T224_25",  (Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_224_res75_50_25/Res025"),), (6, 8), 224),
    "T224_10":  RowSetup("T224_10",  (Path("/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/dinov2_224_res75_50_25/Res010"),), (2, 3), 224),
}

# ==============================================================================
#                            Experiment Descriptors                            
# ==============================================================================

@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    row_ids: Optional[Tuple[str, ...]] = None
    filter_key: Optional[str] = None
    default_train_sizes: Optional[Tuple[int, ...]] = None
    enforce_multiple_of: Optional[int] = None
    cfg_overrides: Optional[Dict[str, object]] = None
    expected_tray_count: Optional[int] = None
    train_images_per_tray: Optional[Tuple[int, ...]] = None
    val_images_per_tray: Optional[Tuple[int, ...]] = None
    mass_feature_mode: Optional[str] = None
    loss_params: Optional[Dict[str, object]] = None

@dataclass(frozen=True)
class StudyPlan:
    name: str
    description: str
    scenarios: Tuple[Scenario, ...]
    requires_row_choice: bool = False
    default_row_id: Optional[str] = None

@dataclass
class SampleTraits:
    sid: int
    tray_ids: set
    moisture_states: set
    capture_count: int
    @property
    def regime(self) -> str:
        count = len(self.tray_ids)
        if count == 1: return "single"
        if count == 4: return "four_tray"
        return "other"
    @property
    def has_moist_and_dry(self) -> bool:
        return {"moist", "dry"}.issubset(self.moisture_states)
    @property
    def has_moist(self) -> bool: return "moist" in self.moisture_states
    @property
    def has_dry(self) -> bool: return "dry" in self.moisture_states

# ==============================================================================
#                           List of Ablation Plans                            
# ==============================================================================

STUDY_PLANS: Dict[str, StudyPlan] = {
    "sample-count": StudyPlan(
        name="sample-count",
        description="Multiples of four up to the available training pool per fold.",
        scenarios=(
            Scenario(key="train_pool_sweep",
                     description="Multiples of four up to the available training pool per fold.",
                     enforce_multiple_of=4,
                     default_train_sizes=(32, 64, 96, 128, 160, 192, 224, 252),),
        ),
        requires_row_choice=True,
        default_row_id="T224",
    ),
    
    "images_per_bag": StudyPlan(
        name="images_per_bag",
        description="Sweep inference tray image counts.",
        scenarios=(
            Scenario(key="single_val_sweep",
                     description="Single-tray samples. Training fixed to 1 image/tray, val sweeps 1-6.",
                     filter_key="single",
                     expected_tray_count=1,
                     train_images_per_tray=(1,),
                     val_images_per_tray=IMAGES_PER_TRAY_SWEEP,),
            Scenario(key="four_tray_val_sweep",
                     description="Four-tray samples. Training fixed to 1 image/tray, val sweeps 1-6.",
                     filter_key="four_tray",
                     expected_tray_count=4,
                     train_images_per_tray=(1,),
                     val_images_per_tray=IMAGES_PER_TRAY_SWEEP,),
        ),
        requires_row_choice=True,
        default_row_id="T224",                 
    ),

    "moisture": StudyPlan(
        name="moisture",
        description="Compare performance on moist-only, dry-only, paired captures, and mass features.",
        scenarios=(
            Scenario(key="dry_only_with_mass",
                     description="Dry captures from paired samples + dry sample mass feature.",
                     filter_key="dry_only", mass_feature_mode="dry",),
            Scenario(key="moist_only_with_mass",
                     description="Moist captures from paired samples + moist sample mass feature.",
                     filter_key="moist_only", mass_feature_mode="moist",),
            Scenario(key="paired_moist_dry_with_mass",
                     description="Moist+dry captures + both masses.",
                     filter_key="moist_and_dry", mass_feature_mode="both",),
            Scenario(key="moist_only",
                     description="Moist captures from samples providing both moisture states.",
                     filter_key="moist_only",),
            Scenario(key="dry_only",
                     description="Dry captures from samples providing both moisture states.",
                     filter_key="dry_only",),
            Scenario(key="paired_moist_dry",
                     description="Moist and dry captures from paired samples.",
                     filter_key="moist_and_dry",),
        ),
        requires_row_choice=True,
        default_row_id="T224",
    ),

    "resolution": StudyPlan(
        name="resolution",
        description="Evaluate four resolution scales (25%, 50%, 75%, 100%) to capture joint effects of spatial detail and bag size.",
        scenarios=(
            Scenario(
                key="resolution_sweep",
                description="Sweep across 25%, 50%, 75%, and 100% resolutions.",
                row_ids=("T224_10",
                         #"T224_25", 
                         #"T224_50", 
                         #"T224_75", 
                         #"T224" #Full resolution
                         ),
            ),
        ),
        requires_row_choice=False, 
    ),

}

def describe_plans() -> str:
    lines: List[str] = []
    for plan_key, plan in STUDY_PLANS.items():
        lines.append(f"Plan '{plan_key}': {plan.description}")
        if plan.requires_row_choice:
            default = f" (default tile row: {plan.default_row_id})" if plan.default_row_id else ""
            lines.append(f"  • Requires --tile-row{default}")
        for scenario in plan.scenarios:
            lines.append(f"    - Scenario '{scenario.key}': {scenario.description}")
            if scenario.cfg_overrides:
                overrides = ", ".join(f"{k}={v}" for k, v in scenario.cfg_overrides.items())
                lines.append(f"      cfg overrides: {overrides}")
    return "\n".join(lines)

def resolve_image_combos(scenario: Scenario) -> List[Optional[Dict[str, object]]]:
    if scenario.expected_tray_count is None: return [None]
    train_opts = scenario.train_images_per_tray or (1,)
    val_opts = scenario.val_images_per_tray or (1,)
    combos: List[Dict[str, object]] = []
    for tr in train_opts:
        if int(tr) <= 0: continue
        for va in val_opts:
            if int(va) <= 0: continue
            combos.append({
                "train_per_tray": int(tr), "val_per_tray": int(va), "expected_tray_count": int(scenario.expected_tray_count), 
                "label": f"trainTray{int(tr):02d}_valTray{int(va):02d}",
            })
    return combos or [None]

# ==============================================================================
#                                   Losses                   
# ==============================================================================

def bin_geom(bin_x: List[float], device: str):
    x = torch.as_tensor(bin_x, dtype=torch.float32, device=device)
    dx = torch.empty_like(x)
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]
    return x, dx

def loss_cvm(cdf_pred: torch.Tensor, cdf_true: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    per_sample = torch.sum((cdf_pred - cdf_true).pow(2) * dx, dim=-1)   
    return torch.mean(per_sample)                                       

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

LOSS_FUNCTIONS: Dict[str, LossFn] = {
    "cvm": loss_cvm,
}

# ==============================================================================
#                               Workers & Seeds                            
# ==============================================================================

cpu_cnt = (os.cpu_count() or 4)
CFG["NUM_WORKERS"] = max(0, min(8, cpu_cnt - 2))
CFG["VAL_WORKERS"] = max(0, min(8, cpu_cnt - 2))
if os.name == "nt":
    CFG["NUM_WORKERS"] = min(CFG["NUM_WORKERS"], 4)
    CFG["VAL_WORKERS"] = min(CFG["VAL_WORKERS"], 4)

def _worker_init_fn(worker_id: int) -> None:
    base = CFG.get("SEED", 1337)
    seed = base + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        if wi is not None and hasattr(wi.dataset, "rng"):
            wi.dataset.rng = np.random.default_rng(seed)
    except Exception: pass

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==============================================================================
#                        PSD and Mass Labels                 
# ==============================================================================

def validate_sample_id_column(df: pd.DataFrame, sample_col: str, context: str,) -> str:
    if sample_col not in df.columns:
        raise ValueError(f"[{context}] Sample ID column '{sample_col}' not found.")
    sids = pd.to_numeric(df[sample_col], errors="coerce")
    bad_ids = sids.isna()
    if bad_ids.any():
        bad_vals = df.loc[bad_ids, sample_col].astype(str).tolist()
        raise ValueError(f"[{context}] Non-numeric sample IDs in '{sample_col}': {bad_vals}.")
    ids_int = sids.astype(int)
    dup_mask = ids_int.duplicated()
    if dup_mask.any():
        dup_vals = sorted(set(ids_int[dup_mask].tolist()))
        raise ValueError(f"[{context}] Duplicate sample IDs: {dup_vals}.")
    df[sample_col] = ids_int
    return sample_col

def load_PSD_labels(xlsx_path: str) -> Tuple[Dict[int, np.ndarray], List[float]]:
    df = pd.read_excel(xlsx_path, usecols="A:R")
    SAMPLE_COL = validate_sample_id_column(df, SAMPLE_ID_COLUMN, context="PSD Excel")

    missing_psd = [col for col in PSD_COLUMNS if col not in df.columns]
    if missing_psd:
        raise ValueError("[PSD Excel] Missing PSD columns: " + ", ".join(missing_psd))
    
    mm_cols = sorted(((mm, col) for col, mm in PSD_COLUMNS.items()), key=lambda t: t[0],)
    cols_ordered = [col for _, col in mm_cols]

    M = df[cols_ordered].apply(_to_float_series, axis=0)
    all_nan = M.isna().all(axis=1)
    any_nan = M.isna().any(axis=1)
    partial = (~all_nan) & any_nan

    keep = ~(all_nan | partial)
    df = df.loc[keep].reset_index(drop=True)
    M = M.loc[keep].reset_index(drop=True)

    BIN_MM = np.array([mm for mm, _ in mm_cols], dtype=np.float32)      
    BIN_X = np.log10(BIN_MM)                                         

    PP = M.to_numpy(dtype=np.float32)  
    CDF = np.nan_to_num(PP / 100.0, nan=0.0).astype(np.float32) 

    SIDS = df[SAMPLE_COL].to_numpy(dtype=int)   
    psd_map = {int(s): CDF[i] for i, s in enumerate(SIDS)}

    non_monotone: List[int] = []
    for sid, cdf in zip(SIDS, CDF):
        if not np.all(cdf[1:] >= cdf[:-1]): 
            non_monotone.append(int(sid))
    if non_monotone:
        ids_preview = ", ".join(map(str, non_monotone))
        raise ValueError(f"[PSD Excel] Non-monotone CDF detected for sample IDs: {ids_preview}.")
    return psd_map, BIN_X.tolist()

def load_sample_masses(xlsx_path: str) -> Dict[int, Dict[str, float]]:
    df = pd.read_excel(xlsx_path)
    SAMPLE_COL = validate_sample_id_column(df, SAMPLE_ID_COLUMN, context="Mass Excel")

    moist_matches = [c for c in df.columns if re.search(MOIST_MASS_COLUMN, str(c), re.I)]
    dry_matches   = [c for c in df.columns if re.search(DRY_MASS_COLUMN,  str(c), re.I)]
    moist_col = moist_matches[0] if moist_matches else None
    dry_col   = dry_matches[0] if dry_matches else None

    if not moist_col and not dry_col: raise ValueError("No mass columns found.")
    df_clean = pd.DataFrame({"sid": df[SAMPLE_COL].astype(int),
                             "moist": _to_float_series(df[moist_col]) if moist_col else None,  
                             "dry": _to_float_series(df[dry_col]) if dry_col else None,})    
    
    mass_records: Dict[int, Dict[str, float]] = {}
    for _, row in df_clean.iterrows():
        sid = int(row["sid"])
        moist_val = row.get("moist")
        dry_val = row.get("dry")
        if pd.notna(moist_val) and pd.notna(dry_val):
            mass_records[sid] = {"moist": float(moist_val), "dry": float(dry_val),}
    return mass_records

def build_mass_feature_map(mass_lookup: Dict[int, Dict[str, float]], mode: str) -> Dict[int, np.ndarray]:
    if mode not in {"moist", "dry", "both"}:
        raise ValueError(f"Unknown mass feature mode '{mode}'.")
    required = ("moist",) if mode == "moist" else ("dry",) if mode == "dry" else ("moist", "dry")

    features: Dict[int, np.ndarray] = {}
    for sid, rec in mass_lookup.items():
        values: List[float] = []
        missing = False
        for key in required:            
            mass_value = rec.get(key)
            if mass_value is None or (isinstance(mass_value, float) and math.isnan(mass_value)):
                missing = True
                break
            values.append(float(mass_value) / MAX_MASS_G)
        if missing: continue
        features[int(sid)] = np.asarray(values, dtype=np.float32)
    return features

# ==============================================================================
#           Image Encodings parsing (tray, moisture)
# ==============================================================================

def _capture_metadata(name: str) -> Tuple[Optional[int], Optional[str]]:
    tray, state = None, None
    name = Path(name).name.lower()
    m = TRAY_RX.search(name)
    if m:
        try: tray = int(m.group(1))
        except ValueError: pass
    if DRY_RX.search(name): state = "dry"
    elif MOIST_RX.search(name): state = "moist"
    return tray, state

def collect_sample_traits(by_sample: Dict[int, List[str]]) -> Dict[int, SampleTraits]:
    traits: Dict[int, SampleTraits] = {}        
    for sid, captures in by_sample.items():
        trays, states = set(), set()
        for cap in captures:
            tray, state = _capture_metadata(cap)
            if tray is not None: trays.add(tray)
            if state: states.add(state)
        traits[sid] = SampleTraits(sid=sid, tray_ids=trays, moisture_states=states, capture_count=len(captures),)
    return traits

def group_captures_by_tray(captures: List[str]) -> Dict[Optional[int], List[str]]:
    groups: Dict[Optional[int], List[str]] = defaultdict(list)
    for cap in captures:
        tray, _ = _capture_metadata(cap)
        groups[tray].append(cap)
    return groups

def filter_captures_by_moisture(captures: List[str], allowed_states: Optional[Collection[str]]) -> List[str]:
    if not allowed_states: return list(captures)
    allowed = {str(state).lower() for state in allowed_states}
    return [cap for cap in captures if _capture_metadata(cap)[1] in allowed]

def select_sample_ids(traits: Dict[int, SampleTraits], filter_key: Optional[str]) -> List[int]:
    if not filter_key: return sorted(traits.keys())
    if filter_key == "single": return sorted([s for s, t in traits.items() if t.regime == "single"])
    if filter_key == "four_tray": return sorted([s for s, t in traits.items() if t.regime == "four_tray"])
    if filter_key in ("moist_and_dry", "moist_only", "dry_only"): 
        return sorted([s for s, t in traits.items() if t.has_moist_and_dry])
    raise ValueError(f"Unknown filter key: {filter_key}")

def resolve_moisture_filter(filter_key: Optional[str]) -> Optional[Tuple[str, ...]]:
    if filter_key == "moist_only": return ("moist",)
    if filter_key == "dry_only": return ("dry",)
    if filter_key == "moist_and_dry": return ("moist", "dry")
    return None

def index_features(root: Path, psd_map: Dict[int, np.ndarray],) -> Tuple[Dict[int, List[str]], List[int]]:
    npz_list = sorted(root.rglob("*.npz"))
    if not npz_list: raise FileNotFoundError(f"No .npz under {root}")

    by_sid: Dict[int, List[str]] = defaultdict(list)
    for p in npz_list:
        m = SID_RX.search(p.name)
        if m: by_sid[int(m.group("sid"))].append(str(p))

    labeled_all = set(int(sid) for sid in psd_map.keys())
    with_npz = set(by_sid.keys())
    missing_npz = sorted(labeled_all - with_npz)
    if missing_npz:
        raise RuntimeError(f"[FATAL] {len(missing_npz)} samples lack .npz files.")

    labeled_sids = sorted(int(sid) for sid in psd_map.keys())
    by_sample: Dict[int, List[str]] = {sid: sorted(by_sid[sid]) for sid in labeled_sids}
    return by_sample, labeled_sids

def fourier_encode_unit(x: np.ndarray, n_freq: int = 8) -> np.ndarray:
    freqs = (2.0 ** np.arange(n_freq, dtype=np.float32)) * np.pi
    xf = x[..., None] * freqs  
    return np.concatenate([np.sin(xf), np.cos(xf)], axis=-1).reshape(x.shape[0], -1)

# ========================== Tray Diagnostics =========================

class ShortageTracker:
    def __init__(self, base_context: Optional[Dict[str, object]] = None) -> None:
        self.base_context: Dict[str, object] = dict(base_context or {})
        self.rows: List[Dict[str, object]] = []
        self._seen: set[Tuple[Optional[str], int, Optional[int], int, str]] = set()
        self.columns: Tuple[str, ...] = ("plan", "scenario", "phase", "images_per_bag", "combo_label", "expected_tray_count",
                                         "images_per_tray", "sample_id", "tray_id", "required_images", "available_images",
                                         "duplicates_added", "status",)

    def record(self, *, sid: int, tray_id: Optional[int], required: int, available: int, duplicates_added: int,
               status: str, context: Optional[Dict[str, object]] = None,) -> None:
        phase = str(self.base_context.get("phase")) if self.base_context else None
        key = (phase, int(sid), int(tray_id) if tray_id is not None else None, int(required), status)
        if key in self._seen: return
        self._seen.add(key)
        row = dict(self.base_context)
        if context: row.update(context)
        row.update({"sample_id": int(sid), "tray_id": int(tray_id) if tray_id is not None else None, "required_images": int(required), 
                    "available_images": int(available), "duplicates_added": int(duplicates_added), "status": status,})
        self.rows.append(row)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows) if self.rows else pd.DataFrame(columns=self.columns)
        for col in self.columns:
            if col not in df.columns: df[col] = None
        df = df.loc[:, self.columns]
        if not df.empty:
            df = df.sort_values(["sample_id", "tray_id", "status"], ignore_index=True)
        df.to_csv(path, index=False)

# --------------------- MIL Bag Construction ----------------------

class BagDataset(torch.utils.data.Dataset):
    def __init__(self, sids, by_sample, psd_map, k_imgs=4, seed=0, pos_fourier=None,
                 eval_mode=False, add_global_tile: bool = True, images_per_tray: Optional[int] = None,
                 expected_tray_count: Optional[int] = None, shortage_tracker: Optional[ShortageTracker] = None,
                 phase: str = "train", regime_label: Optional[str] = None, context: Optional[Dict[str, object]] = None,
                 max_extra_per_image: int = 1, moisture_filter: Optional[Collection[str]] = None,
                 sample_features: Optional[Dict[int, np.ndarray]] = None,):
        self.sids = list(sids)
        self.by_sample = self._build_filtered_by_sample(by_sample, moisture_filter)
        self.psd_map = psd_map
        self.k = int(k_imgs)
        self.rng = np.random.default_rng(seed)
        self.pos_fourier = CFG.get("POS_FOURIER", 8) if pos_fourier is None else pos_fourier
        self.eval_mode = eval_mode
        self.seed_base = int(seed)
        self.add_global_tile = bool(add_global_tile)
        self.capture_groups = {sid: group_captures_by_tray(captures) for sid, captures in self.by_sample.items()}
        self._init_tray_budgets(images_per_tray, expected_tray_count)
        self.shortage_tracker = shortage_tracker
        self.phase = str(phase)
        self.regime_label = regime_label
        self.context_info: Dict[str, object] = dict(context or {})
        self.max_extra_per_image = max(0, int(max_extra_per_image))
        self._init_sample_features(sample_features)

    def _build_filtered_by_sample(self, by_sample: Dict[int, List[str]], moisture_filter: Optional[Collection[str]],) -> Dict[int, List[str]]:
        allowed_states = {str(s).lower() for s in moisture_filter} if moisture_filter else None
        subset, filtered_by_sample = {}, {}
        for sid in self.sids:
            if sid not in by_sample: raise KeyError(f"Sample ID {sid} not found.")
            subset[sid] = by_sample[sid]
        for sid, captures in subset.items():
            filtered = filter_captures_by_moisture(captures, allowed_states)
            if allowed_states and not filtered:
                raise RuntimeError(f"Sample ID {sid} has no matching captures.")
            filtered_by_sample[sid] = filtered
        return filtered_by_sample

    def _init_tray_budgets(self, images_per_tray: Optional[int], expected_tray_count: Optional[int]) -> None:
        self.images_per_tray = int(images_per_tray) if images_per_tray is not None else None
        self.expected_tray_count = int(expected_tray_count) if expected_tray_count is not None else None
        if (self.images_per_tray is None) != (self.expected_tray_count is None):
            raise ValueError("images_per_tray and expected_tray_count must be provided together.")
        if self.images_per_tray is not None:
            if self.images_per_tray <= 0: raise ValueError("images_per_tray must be positive.")
            if self.expected_tray_count is None or self.expected_tray_count <= 0:
                raise ValueError("expected_tray_count must be positive.")
            self.k = self.images_per_tray * self.expected_tray_count

    def _init_sample_features(self, sample_features: Optional[Dict[int, np.ndarray]]) -> None:
        self.sample_features: Dict[int, np.ndarray] = {}
        self.sample_feature_dim = 0
        if sample_features:
            dims = set()
            for sid_raw, vec in sample_features.items():
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.size == 0: continue
                self.sample_features[int(sid_raw)] = arr
                dims.add(arr.size)
            if dims:
                if len(dims) != 1: raise ValueError("Sample features must share the same length.")
                self.sample_feature_dim = dims.pop()
        self.require_sample_features = bool(sample_features)
        if self.require_sample_features:
            missing = [sid for sid in self.sids if sid not in self.sample_features]
            if missing: raise KeyError(f"Sample features missing for {len(missing)} SIDs.")    

    def __len__(self): return len(self.sids)

    def _record_shortage(self, *, sid: int, tray_id: Optional[int], required: int, available: int, duplicates_added: int, status: str) -> None:
        if not self.shortage_tracker: return
        context = dict(self.context_info)
        if "N" in context: context["train_size"] = context.pop("N")
        if self.regime_label is not None and "images_per_bag" not in self.shortage_tracker.base_context:
            context.setdefault("images_per_bag", self.regime_label)
        self.shortage_tracker.record(sid=sid, tray_id=tray_id, required=required, available=available, 
                                     duplicates_added=duplicates_added, status=status, context=context)
    
    def _select_from_tray(self, sid: int, tray_id: Optional[int], options: List[str], required: int, deterministic: bool,) -> List[str]:
        local = list(options)
        if not local:
            self._record_shortage(sid=sid, tray_id=tray_id, required=required, available=0, duplicates_added=required, status="empty_tray")
            raise RuntimeError(f"No captures available for SID {sid} tray {tray_id}")
        if deterministic: local.sort()
        else: self.rng.shuffle(local)

        available = len(local)
        if available >= required: return local[:required]

        shortage = required - available
        self._record_shortage(sid=sid, tray_id=tray_id, required=required, available=available, duplicates_added=shortage, status="reused")
        max_allowed = available * (1 + self.max_extra_per_image)
        if required > max_allowed:
            self._record_shortage(sid=sid, tray_id=tray_id, required=required, available=available, duplicates_added=required - available, status="unsatisfiable")
            raise RuntimeError(f"Cannot satisfy tray requirement for SID {sid} tray {tray_id}.")

        duplicate_pool = list(local)
        if deterministic: duplicates = duplicate_pool[:shortage]
        else:
            self.rng.shuffle(duplicate_pool)
            duplicates = duplicate_pool[:shortage]
        selection = local + duplicates
        if not deterministic: self.rng.shuffle(selection)
        return selection

    def _select_by_tray(self, sid: int, groups: Dict[Optional[int], List[str]], ) -> List[str]:
        tray_ids = [t for t in groups.keys() if t is not None]
        tray_ids_sorted = sorted(tray_ids)
        if len(tray_ids_sorted) != self.expected_tray_count:
            raise RuntimeError(f"SID {sid} has {len(tray_ids_sorted)} trays, expected {self.expected_tray_count}")
        tray_sequence = tray_ids_sorted[:]
        if not self.eval_mode: self.rng.shuffle(tray_sequence)

        chosen: List[str] = []
        for tray_id in tray_sequence:
            chosen.extend(self._select_from_tray(sid, tray_id, groups[tray_id], self.images_per_tray, self.eval_mode))
        if len(chosen) != self.images_per_tray * len(tray_sequence):
            raise RuntimeError("Tray selection mismatch.")
        return chosen

    def _tile_features(self, d) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        cls = d["cls"].astype(np.float32)     
        mean = d["mean"].astype(np.float32)    
        coords = d["coords"].astype(np.float32) 
        meta = json.loads(str(d["meta"].item()) if hasattr(d["meta"], "item") else str(d["meta"]))
        
        # FIX: Parse w and h from the "cropped_size" string (e.g., "7040x5184")
        w_str, h_str = meta["cropped_size"].split("x")
        w, h = float(w_str), float(h_str)
        
        rows, cols = int(meta["rows"]), int(meta["cols"])

        x0, y0, x1, y1 = [coords[:, i] for i in range(4)]
        ww, hh = np.maximum(1.0, x1 - x0), np.maximum(1.0, y1 - y0)
        cx, cy = (x0 + ww * 0.5) / max(w, 1.0), (y0 + hh * 0.5) / max(h, 1.0)
        sx, sy = ww / max(w, 1.0), hh / max(h, 1.0)
        pos = np.stack([cx, cy, sx, sy], axis=-1)  
        if self.pos_fourier and self.pos_fourier > 0:
            pos = fourier_encode_unit(pos.astype(np.float32), n_freq=self.pos_fourier)

        g = d["global_cls"].astype(np.float32)               
        g_broadcast = np.repeat(g[None, :], cls.shape[0], axis=0)           
        X = np.concatenate([cls, mean, g_broadcast, pos.astype(np.float32)], axis=1)  

        return X, np.arange(cls.shape[0], dtype=np.int32), g, {"rows": rows, "cols": cols}

    def __getitem__(self, idx: int):
        sid = int(self.sids[idx])
        captures = self.by_sample[sid]
        groups = self.capture_groups.get(sid, {})
        extra_vec = self.sample_features.get(sid) if self.sample_feature_dim else None
        tray_keys = [t for t in groups.keys() if t is not None]

        sel = self._select_captures_for_sid(sid, captures, groups, tray_keys)
        return self._build_bag_from_captures(sid, sel, extra_vec)

    def _select_captures_for_sid(self, sid: int, captures: List[str], groups: Dict[Optional[int], List[str]], tray_keys: List[Optional[int]]) -> List[str]:
        if self.images_per_tray is not None and self.expected_tray_count is not None:
            return self._select_by_tray(sid, groups)

        if getattr(self, "eval_mode", False):
            ordered: List[str] = []
            for tray in sorted(tray_keys): ordered.extend(sorted(groups[tray]))
            if None in groups: ordered.extend(sorted(groups[None]))
            if not ordered: ordered = sorted(captures)
            if len(ordered) >= self.k: return ordered[: self.k]
            return (ordered * ((self.k + len(ordered) - 1) // max(1, len(ordered))))[: self.k]

        sel: List[str] = []
        if tray_keys:
            self.rng.shuffle(tray_keys)
            for tray in tray_keys:
                if len(sel) >= self.k: break
                sel.append(groups[tray][int(self.rng.integers(len(groups[tray])))])

        if len(sel) < self.k:
            remaining = self.k - len(sel)
            if not captures: raise RuntimeError(f"No captures for SID {sid}")
            idxs = np.atleast_1d(self.rng.choice(len(captures), size=remaining, replace=len(captures) < remaining))
            for j in idxs: sel.append(captures[int(j)])
        return sel

    def _build_bag_from_captures(self, sid: int, sel: List[str], extra_vec: Optional[np.ndarray]) -> Dict[str, object]:
        bag_parts, img_idx_all, tile_idx_all = [], [], []
        rows_cols: Dict[int, Tuple[int, int]] = {}

        for img_i, path in enumerate(sel):
            try:
                with np.load(str(path), allow_pickle=True) as d:
                    X_local, tile_idx_arr, g_vec, info = self._tile_features(d)
            except Exception as exc:
                raise RuntimeError(f"Failed to open feature archive {path}: {exc}") from exc

            rows_cols[img_i] = (info.get("rows", 8), info.get("cols", 10))
            g_cap = g_vec.astype(np.float32)
            g_cap = g_cap / max(np.linalg.norm(g_cap), 1e-8)

            X_aug = X_local
            if self.add_global_tile:
                zeros_pos = np.zeros((1, X_local.shape[1] - g_cap.size * 3), dtype=np.float32)
                g_tile = np.concatenate([g_cap, g_cap, g_cap, zeros_pos[0]], axis=0)[None, :]
                X_aug = np.vstack([X_local, g_tile])
                tile_idx_arr = np.concatenate([tile_idx_arr, np.array([-1], dtype=np.int32)])

            if self.sample_feature_dim and extra_vec is not None:
                extras = np.repeat(extra_vec[None, :], X_aug.shape[0], axis=0)
                X_aug = np.concatenate([X_aug, extras.astype(np.float32)], axis=1)

            X_all = X_aug.astype(np.float32)
            bag_parts.append(X_all)
            tile_idx_all.append(tile_idx_arr)
            img_idx_all.append(np.full((X_all.shape[0],), img_i, dtype=np.int32))

        return {
            "x": torch.from_numpy(np.concatenate(bag_parts, axis=0).astype(np.float32)), 
            "y": torch.from_numpy(self.psd_map[sid].astype(np.float32)), 
            "sid": sid, 
            "meta": {"rows_cols": {int(k): tuple(v) for k, v in rows_cols.items()},
                     "img_idx": np.concatenate(img_idx_all).tolist(),
                     "tile_idx": np.concatenate(tile_idx_all).tolist(),
                     "num_images": self.k,}
        }

def collate(items):
    return {"x": torch.stack([it["x"] for it in items], 0), 
            "y": torch.stack([it["y"] for it in items], 0), 
            "sid": [it["sid"] for it in items], 
            "meta": [it["meta"] for it in items]}

def make_loader(sids, by_sample, psd_map, batch, shuffle, workers, bag_k, seed, add_global_tile: bool = True, 
                images_per_tray: Optional[int] = None, expected_tray_count: Optional[int] = None, shortage_tracker: Optional[ShortageTracker] = None, 
                phase: str = "train", regime: Optional[str] = None, context: Optional[Dict[str, object]] = None, max_extra_per_image: int = 1, 
                moisture_filter: Optional[Collection[str]] = None, sample_features: Optional[Dict[int, np.ndarray]] = None,):
    ds = BagDataset(sids, by_sample, psd_map, k_imgs=bag_k, seed=seed, eval_mode=not shuffle, 
                    add_global_tile=add_global_tile, images_per_tray=images_per_tray, expected_tray_count=expected_tray_count, 
                    shortage_tracker=shortage_tracker, phase=phase, regime_label=regime, context=context, 
                    max_extra_per_image=max_extra_per_image, moisture_filter=moisture_filter, sample_features=sample_features,)
    bs = min(batch, max(1, len(sids)))
    kwargs = dict(batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(), collate_fn=collate, 
                  worker_init_fn=_worker_init_fn, prefetch_factor=2 if workers > 0 else None,)
    if kwargs["prefetch_factor"] is None: kwargs.pop("prefetch_factor")
    return DataLoader(ds, **kwargs)

# ---------------------- Model Architecture ---------------------

class SimpleMILHead(nn.Module):
    def __init__(self, d_in: int, n_bins: int, d_att: int = 256, d_mlp: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.attn_net = nn.Sequential(nn.Linear(d_in, d_att), nn.Tanh(), nn.Linear(d_att, 1))
        self.proj = nn.Sequential(nn.Linear(d_in, d_mlp), nn.ReLU(True), nn.Dropout(p_drop), nn.Linear(d_mlp, n_bins))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, D = x.shape
        a = self.attn_net(x).squeeze(-1)      
        attn = torch.softmax(a, dim=1)        
        z = torch.bmm(attn.unsqueeze(1), x).squeeze(1)   
        logits = self.proj(z)                  
        pdf = torch.softmax(logits, dim=-1)
        cdf = torch.cumsum(pdf, dim=-1).clamp(max=1.0)
        return {"cdf": cdf, "pdf": pdf, "attn": attn, "scale_weights": torch.ones(B, 1, device=x.device, dtype=x.dtype)}

# =========================== Train and Evaluation ==========================

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, x_geom, device: str, loss_fn: LossFn = loss_cvm,) -> float:
    model.eval()
    _, dx = x_geom
    tot, n = 0.0, 0
    for b in loader:
        xb = b["x"].to(device, non_blocking=True)
        yb = b["y"].to(device, non_blocking=True)
        loss = loss_fn(model(xb)["cdf"], yb.to(torch.float32), dx)
        bs = xb.size(0)
        tot += float(loss) * bs
        n += bs
    return tot / max(1, n)

@torch.inference_mode()
def save_fold_predictions(model: nn.Module, loader: DataLoader, x_geom, device: str, path: Path) -> None:
    model.eval()
    x, _ = x_geom
    bin_x = x.detach().cpu().numpy()
    preds: List[Dict] = []
    for b in loader:
        xb = b["x"].to(device, non_blocking=True)
        yb = b["y"].to(device, non_blocking=True)
        sids = b["sid"]
        out = model(xb)
        pred, true = out["cdf"].detach().cpu().numpy(), yb.to(torch.float32).detach().cpu().numpy()
        for sid, pcdf, tcdf in zip(sids, pred, true):
            preds.append({"sid": int(sid), "pred_cdf": pcdf.tolist(), "true_cdf": tcdf.tolist(),})
    with open(path, "w") as f:
        json.dump({"bin_x_log10": bin_x.tolist(), "predictions": preds}, f)

def train_one(model: nn.Module, tr_loader: DataLoader, va_loader: DataLoader, x_geom, epochs: int, patience: int, 
              device: str, writer: SummaryWriter = None, tag: str = None, loss_fn: LossFn = loss_cvm,):
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warm = max(2, epochs // 10)  
    sched = SequentialLR(opt, schedulers=[LinearLR(opt, start_factor=0.1, total_iters=warm), CosineAnnealingLR(opt, T_max=max(1, epochs - warm))], milestones=[warm])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["USE_AMP"])
    best = {"loss": float("inf"), "state": None, "epoch": -1}
    last_improve = 0
    _, dx = x_geom

    for epoch in range(1, epochs + 1):
        model.train()
        s = time.time()
        acc, n = 0.0, 0
        for step, b in enumerate(tr_loader):
            xb, yb = b["x"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=CFG["USE_AMP"]):
                loss = loss_fn(model(xb)["cdf"], yb.to(torch.float32), dx)

            opt.zero_grad(set_to_none=True)
            if CFG["USE_AMP"]:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = xb.size(0)
            acc += float(loss) * bs
            n += bs

        tr_loss = acc / max(1, n)
        va_loss = evaluate(model, va_loader, x_geom, device, loss_fn=loss_fn)

        if writer and tag:
            writer.add_scalar(f"{tag}/loss_train", tr_loss, epoch)
            writer.add_scalar(f"{tag}/loss_val", va_loss, epoch)
            writer.add_scalar(f"{tag}/lr", opt.param_groups[0]["lr"], epoch)

        sched.step()
        dt = time.time() - s
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[{tag}] epoch {epoch:03d}  train {tr_loss:.4f}  val {va_loss:.4f}  ({dt:.1f}s)")

        if va_loss + 1e-8 < best["loss"]:
            best.update({"loss": va_loss, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch})
            last_improve = epoch
        if epoch - last_improve >= patience:
            print(f"[{tag}] early stop @ {epoch} (best {best['epoch']} | {best['loss']:.4f})")
            break

    if best["state"] is not None: model.load_state_dict(best["state"])
    return best

# ================== Split Building & Utilities ==================

def make_folds(sids: List[int], k: int, seed: int) -> List[List[int]]:
    s = sids[:]
    random.Random(seed).shuffle(s)
    return [s[i::k] for i in range(k)]

def build_master_split(sample_ids: List[int], n_folds: int, base_seed: int, run_dir: Path, train_sizes: Optional[Sequence[int]] = None, ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], List[int]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sids = sorted(sample_ids)
    rng = random.Random(base_seed)
    rng.shuffle(sids)
    folds = [sids[i::n_folds] for i in range(n_folds)]
    folds_df = pd.DataFrame([{"sample_id": sid, "fold_id": i} for i, fold in enumerate(folds) for sid in fold])
    folds_df.to_csv(run_dir / "folds.csv", index=False)

    tr_pools: Dict[int, pd.DataFrame] = {}
    for i in range(n_folds):
        tr_ids = sorted(folds_df.loc[folds_df["fold_id"] != i, "sample_id"].tolist())
        random.Random(base_seed).shuffle(tr_ids)
        tr_pools[i] = pd.DataFrame({"sample_id": tr_ids, "fold_id": i, "rank_in_trpool": np.arange(1, len(tr_ids) + 1, dtype=int),})
        tr_pools[i].to_csv(run_dir / f"TRPOOL_{i}.csv", index=False)

    n_max = min(len(df) for df in tr_pools.values())
    if train_sizes:
        cleaned = sorted({int(n) for n in train_sizes if int(n) > 0})
        N_list = [n for n in cleaned if n <= n_max]
        if not N_list or N_list[-1] != n_max: N_list.append(n_max)
    else: N_list = [n_max]

    idx_rows = []
    for i, df in tr_pools.items():
        for _, r in df.iterrows():
            row = {"sample_id": int(r.sample_id), "fold_id": i, "rank_in_trpool": int(r.rank_in_trpool)}
            for N in N_list: row[f"included_at_{N}"] = int(r.rank_in_trpool <= N)
            idx_rows.append(row)
    pd.DataFrame(idx_rows).to_csv(run_dir / "global_index.csv", index=False)
    return folds_df, tr_pools, N_list

def infer_dims(sample_ids: List[int], by_sample, psd_map, bag_k: int, seed: int, add_global_tile: bool, 
               moisture_filter: Optional[Collection[str]] = None, sample_features: Optional[Dict[int, np.ndarray]] = None,) -> Tuple[int, int]:
    ds = BagDataset(sample_ids[:1], by_sample, psd_map, k_imgs=bag_k, seed=seed, add_global_tile=add_global_tile, moisture_filter=moisture_filter, sample_features=sample_features,)
    b = ds[0]
    return b["x"].shape[-1], int(b["y"].numel())

# --------------------- Ablation Runner ----------------------

def validate_plan_setup(plan: StudyPlan, tile_row: Optional[str], train_sizes: Optional[Sequence[int]],) -> str:
    chosen_row = tile_row
    if plan.requires_row_choice:
        chosen_row = chosen_row or plan.default_row_id
        if chosen_row is None or chosen_row not in ROW_SETUPS:
            valid = ", ".join(sorted(ROW_SETUPS))
            raise ValueError(f"[FATAL] Requires --tile-row in {{{valid}}}. Got {chosen_row!r}.")
    elif chosen_row is not None and chosen_row not in ROW_SETUPS:
        valid = ", ".join(sorted(ROW_SETUPS))
        raise ValueError(f"[FATAL] Unknown --tile-row {chosen_row!r}. Choices: {valid}.")

    if not Path(TRUTH_XLSX).exists(): raise FileNotFoundError(f"[FATAL] PSD_TRUTH_XLSX not found: {TRUTH_XLSX}")
    if not FEAT_ROOT.exists(): raise FileNotFoundError(f"[FATAL] FEAT_ROOT not found: {FEAT_ROOT}")

    for scenario in plan.scenarios:
        row_ids = scenario.row_ids or ((chosen_row,) if chosen_row is not None else tuple(ROW_SETUPS.keys()))
        if not row_ids: raise ValueError(f"Scenario '{scenario.key}' does not define any row ids.")
        sizes_override = train_sizes if train_sizes is not None else scenario.default_train_sizes
        if scenario.enforce_multiple_of and sizes_override:
            invalid = [n for n in sizes_override if n % scenario.enforce_multiple_of != 0]
            if invalid: raise ValueError(f"Scenario expects train sizes to be multiples of {scenario.enforce_multiple_of}. Invalid: {invalid}")
    return chosen_row

def run_ablation_study(plan_name: str = "sample-count", tile_row: Optional[str] = None, base_seed: int = CFG["SEED"], repeats: int = REPEATS, train_sizes: Optional[Sequence[int]] = None,) -> None:
    import torch.multiprocessing as mp
    mp.freeze_support()

    plan = STUDY_PLANS[plan_name]
    set_all_seeds(base_seed)
    chosen_row = validate_plan_setup(plan=plan, tile_row=tile_row, train_sizes=train_sizes)
    psd_map, BIN_X = load_PSD_labels(TRUTH_XLSX)

    stamp = (datetime.now()-MONTREAL_OFFSET).strftime("%Y-%m-%d_%H-%M-%S")
    out_root = OUT_BASE / f"{plan.name}_ablation_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    plan_records: List[Dict[str, object]] = []
    mass_lookup_cache: Optional[Dict[int, Dict[str, float]]] = None

    print(f"[INFO] Outputs will be saved to: {out_root}")
    print(f"[START] plan={plan.name} | K={CFG['K_FOLDS']} | EPOCHS={CFG['EPOCHS']} | REPEATS={repeats}")

    for scenario in plan.scenarios:
        row_ids = scenario.row_ids or ((chosen_row,) if chosen_row else tuple(ROW_SETUPS.keys()))
        sizes_override = train_sizes if train_sizes is not None else scenario.default_train_sizes
        for row_id in row_ids:
            row = ROW_SETUPS[row_id]
            mass_lookup_cache = _run_scenario_for_row(plan=plan, scenario=scenario, row=row, row_id=row_id, psd_map=psd_map, 
                                                      BIN_X=BIN_X, out_root=out_root, stamp=stamp, base_seed=base_seed,
                                                      repeats=repeats, sizes_override=sizes_override, mass_lookup_cache=mass_lookup_cache, plan_records=plan_records,)

    (out_root / "plan_manifest.json").write_text(json.dumps({"plan": plan.name, "plan_description": plan.description, "base_seed": base_seed, "repeats": repeats, "timestamp": stamp, "scenarios": plan_records,}, indent=2))

def _run_scenario_for_row(plan: StudyPlan, scenario: Scenario, row: RowSetup, row_id: str, psd_map: Dict[int, np.ndarray],
                          BIN_X: np.ndarray, out_root: Path, stamp: str, base_seed: int, repeats: int,
                          sizes_override: Optional[Sequence[int]], mass_lookup_cache: Optional[Dict[int, Dict[str, float]]], plan_records: List[Dict[str, object]],) -> Optional[Dict[int, Dict[str, float]]]:
    print(f"\n[SCENARIO] {scenario.key} | row={row_id} | description={scenario.description}")
    by_sample, labeled_sids = index_features(row.feat_roots[0], psd_map)
    if not labeled_sids: return mass_lookup_cache

    traits = collect_sample_traits(by_sample)
    sample_ids = select_sample_ids(traits, scenario.filter_key)
    if not sample_ids: return mass_lookup_cache

    scenario_sample_features: Optional[Dict[int, np.ndarray]] = None
    sample_feature_dim = 0
    if scenario.mass_feature_mode:
        if mass_lookup_cache is None: mass_lookup_cache = load_sample_masses(MASS_XLSX)
        feature_lookup = build_mass_feature_map(mass_lookup_cache, scenario.mass_feature_mode)
        available = sorted([sid for sid in sample_ids if sid in feature_lookup])
        sample_ids = available
        if not sample_ids: return mass_lookup_cache
        scenario_sample_features = {sid: feature_lookup[sid] for sid in sample_ids}
        sample_feature_dim = len(next(iter(scenario_sample_features.values())))

    moisture_filter = resolve_moisture_filter(scenario.filter_key)
    scenario_root = out_root / scenario.key / row_id
    splits_root = scenario_root / "splits"
    folds_df, tr_pools, N_list = build_master_split(sample_ids, CFG["K_FOLDS"], base_seed, splits_root, train_sizes=sizes_override)

    if scenario.enforce_multiple_of:
        mult = scenario.enforce_multiple_of
        N_list = [n for n in sorted({n if n % mult == 0 else (n // mult) * mult for n in N_list if n >= mult}) if n > 0]

    cfg_overrides = scenario.cfg_overrides or {}
    loss_key = "cvm"
    loss_fn = LOSS_FUNCTIONS[loss_key]
    add_global_tile = cfg_overrides.get("ADD_GLOBAL_TILE", CFG["ADD_GLOBAL_TILE"])

    scenario_record: Dict[str, object] = {"plan": plan.name, "scenario": scenario.key, "row_id": row_id, "tile_px": row.px,
                                          "tiles_hw": row.tiles_hw, "cfg_overrides": dict(cfg_overrides), "add_global_tile": add_global_tile, 
                                          "loss_key": loss_key, "sample_count": len(sample_ids), "train_sizes": list(N_list), 
                                          "k_folds": CFG["K_FOLDS"], "repeats": repeats, "timestamp": stamp,}
    if moisture_filter: scenario_record["moisture_filter"] = list(moisture_filter)
    if scenario.mass_feature_mode: scenario_record.update({"sample_feature_mode": scenario.mass_feature_mode, "sample_feature_dim": int(sample_feature_dim)})
    if scenario.expected_tray_count is not None: scenario_record["expected_tray_count"] = int(scenario.expected_tray_count)
    if scenario.train_images_per_tray: scenario_record["train_images_per_tray"] = [int(x) for x in scenario.train_images_per_tray]
    if scenario.val_images_per_tray: scenario_record["val_images_per_tray"] = [int(x) for x in scenario.val_images_per_tray]

    scenario_root.mkdir(parents=True, exist_ok=True)
    (scenario_root / "manifest.json").write_text(json.dumps(scenario_record, indent=2))
    plan_records.append(scenario_record)

    _run_training_for_row(plan=plan, scenario=scenario, row=row, row_id=row_id, sample_ids=sample_ids, 
                          scenario_sample_features=scenario_sample_features, moisture_filter=moisture_filter, folds_df=folds_df,
                          tr_pools=tr_pools, N_list=N_list, loss_key=loss_key, loss_fn=loss_fn, add_global_tile=add_global_tile,
                          BIN_X=BIN_X, psd_map=psd_map, scenario_root=scenario_root, repeats=repeats, base_seed=base_seed, by_sample=by_sample,)
    return mass_lookup_cache

def _run_training_for_row(plan: StudyPlan, scenario: Scenario, row: RowSetup, row_id: str, sample_ids: Sequence[int], 
                          scenario_sample_features: Optional[Dict[int, np.ndarray]], moisture_filter: Optional[Sequence[str]],
                          folds_df: "pd.DataFrame", tr_pools: Mapping[int, "pd.DataFrame"], N_list: Sequence[int], loss_key: str,
                          loss_fn, add_global_tile: bool, BIN_X: np.ndarray, psd_map: Dict[int, np.ndarray], scenario_root: Path,
                          repeats: int, base_seed: int, by_sample: Dict[int, List[str]],) -> None:
    image_combos = resolve_image_combos(scenario) if plan.name == "images_per_bag" else [None]
    d_in, n_bins = infer_dims(sample_ids, by_sample=by_sample, psd_map=psd_map, bag_k=CFG["BAG_K"], seed=base_seed,
                              add_global_tile=add_global_tile, moisture_filter=moisture_filter, sample_features=scenario_sample_features,)

    feat_root = row.feat_roots[0]
    x_geom = bin_geom(BIN_X, CFG["DEVICE"])

    for combo in image_combos:
        if combo is None:
            combo_label, combo_root, expected_tray_count, train_images_per_tray, val_images_per_tray = None, scenario_root, None, None, None
            train_bag_k, val_bag_k = CFG["BAG_K"], CFG["BAG_K"]
            train_tracker, val_tracker = None, None
        else:
            combo_label = str(combo.get("label"))
            combo_root = scenario_root / combo_label
            expected_tray_count = int(combo.get("expected_tray_count", 0))
            train_images_per_tray, val_images_per_tray = int(combo.get("train_per_tray", 0)), int(combo.get("val_per_tray", 0))
            train_bag_k, val_bag_k = train_images_per_tray * expected_tray_count, val_images_per_tray * expected_tray_count

            train_tracker = ShortageTracker({"plan": plan.name, "scenario": scenario.key, "phase": "train", "images_per_bag": scenario.filter_key, "combo_label": combo_label, "expected_tray_count": expected_tray_count, "images_per_tray": train_images_per_tray,})
            val_tracker = ShortageTracker({"plan": plan.name, "scenario": scenario.key, "phase": "val", "images_per_bag": scenario.filter_key, "combo_label": combo_label, "expected_tray_count": expected_tray_count, "images_per_tray": val_images_per_tray,})

        combo_root.mkdir(parents=True, exist_ok=True)

        for N in N_list:
            for r in range(repeats):
                rep_seed = base_seed + r
                set_all_seeds(rep_seed)

                run_dir = combo_root / f"N_{N}" / f"seed_{base_seed}" / f"repeat_{r}"
                run_dir.mkdir(parents=True, exist_ok=True)

                cfg = CFG.copy()
                cfg.update(scenario.cfg_overrides or {})
                cfg.update({"FEAT_ROOT_LAST": str(feat_root), "N": N, "row_id": row_id, "base_seed": base_seed, "repeat_id": r, "plan": plan.name, "scenario": scenario.key,})

                if scenario.mass_feature_mode: cfg.update({"sample_feature_mode": scenario.mass_feature_mode, "sample_feature_dim": (len(next(iter(scenario_sample_features.values()))) if scenario_sample_features else 0)})
                if expected_tray_count is not None: cfg.update({"bag_k_train": train_bag_k, "bag_k_val": val_bag_k, "combo_label": combo_label,})
                else: cfg.update({"bag_k_train": CFG["BAG_K"], "bag_k_val": CFG["BAG_K"], "combo_label": combo_label,})

                cfg["LOSS"] = loss_key
                (run_dir / "cfg.json").write_text(json.dumps(cfg, indent=2))

                tb_dir = run_dir / "tb"
                writer = SummaryWriter(log_dir=str(tb_dir)) if CFG["USE_TB"] else None

                fold_losses: List[float] = []
                for fold in range(CFG["K_FOLDS"]):
                    va_ids = folds_df.loc[folds_df["fold_id"] == fold, "sample_id"].tolist()
                    tr_ids = tr_pools[fold].head(N).loc[:, "sample_id"].tolist()

                    model = SimpleMILHead(d_in=d_in, n_bins=n_bins).to(CFG["DEVICE"])

                    tr_loader = make_loader(tr_ids, by_sample, psd_map, CFG["BATCH_SIZE"], True, CFG["NUM_WORKERS"], train_bag_k, rep_seed, add_global_tile=add_global_tile, images_per_tray=train_images_per_tray if expected_tray_count is not None else None, expected_tray_count=expected_tray_count, shortage_tracker=train_tracker, phase="train", regime=scenario.filter_key, context={"N": N, "repeat": r, "combo_label": combo_label}, moisture_filter=moisture_filter, sample_features=scenario_sample_features,)
                    va_loader = make_loader(va_ids, by_sample, psd_map, CFG["BATCH_SIZE"], False, CFG["VAL_WORKERS"], val_bag_k, rep_seed, add_global_tile=add_global_tile, images_per_tray=val_images_per_tray if expected_tray_count is not None else None, expected_tray_count=expected_tray_count, shortage_tracker=val_tracker, phase="val", regime=scenario.filter_key, context={"N": N, "repeat": r, "combo_label": combo_label}, moisture_filter=moisture_filter, sample_features=scenario_sample_features,)

                    best = train_one(model, tr_loader, va_loader, x_geom, CFG["EPOCHS"], CFG["PATIENCE"], CFG["DEVICE"], writer, tag=f"fold_{fold+1}", loss_fn=loss_fn, )
                    fold_losses.append(best["loss"])
                    save_fold_predictions(model, va_loader, x_geom, CFG["DEVICE"], run_dir / f"fold_{fold+1}_preds.json")

                with open(run_dir / "kfold_summary.json", "w") as f:
                    json.dump({"fold_val": fold_losses, "mean": float(np.mean(fold_losses)), "std": float(np.std(fold_losses)), "k": CFG["K_FOLDS"], "N": N, "row_id": row_id, "plan": plan.name, "scenario": scenario.key, "loss": loss_key, "combo_label": combo_label,}, f, indent=2)

                if writer: writer.close()

        if train_tracker: train_tracker.write(combo_root / "train_shortages.csv")
        if val_tracker: val_tracker.write(combo_root / "val_shortages.csv")

# ==============================================================================
#                                     MAIN                            
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated MIL ablation plans.")
    parser.add_argument("--plan", choices=sorted(STUDY_PLANS.keys()), default="sample-count", help="Select which ablation plan to execute.",)
    
    args = parser.parse_args()

    print(f"[INFO] Effective workers: NUM_WORKERS={CFG['NUM_WORKERS']}, VAL_WORKERS={CFG['VAL_WORKERS']}")

    run_ablation_study(
        plan_name=args.plan, 
        tile_row="T224", 
        base_seed=CFG["SEED"], 
        repeats=REPEATS, 
        train_sizes=None,
    )

if __name__ == "__main__":
    main()