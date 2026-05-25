import os
# Ensure legacy routing is active
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
# Configure thread usage for 70 cores
tf.config.threading.set_intra_op_parallelism_threads(68) 
tf.config.threading.set_inter_op_parallelism_threads(2)

import tf_keras
import h5py

print('TensorFlow version:', tf.__version__)
print('tf-keras (Legacy) version:', tf_keras.__version__)
print('h5py version:', h5py.__version__)

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from train_test import run_train, run_test
from test_vis import create_plots
from inference_bank import run_prediction_orthophoto, read_rgb
from helper import setup_parser, collect_cv_data, create_k_fold_split_indices, calculate_curve_mae_percent

# setup argument parser with default values
parser = setup_parser()
args, unknown = parser.parse_known_args()

# Use your PSD-based dataset
args.data_npz_path = "/home/thomas_plante_stcyr/workspace/torch/2_1_1/scratch/extracted_features/grainet_data_global_from_PSD_500x200.npz"
                     
# Not using orthophoto prediction yet
args.image_path = None
ortho_mask_path = None

# Output directory for this experiment
parent_dir = "output_global_from_PSD_dm_500x200"

metrics_keys = ('mae', 'rmse')

# Keep dm mode (simplest): predict scalar dm
args.output_dm = False
args.loss_key = 'mse'    # dm regression

args.verbose = 1
args.nb_epoch = 20

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# print all arguments
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

import numpy as np
import os
import re

num_folds = 5
data = np.load(args.data_npz_path, allow_pickle=True)
names = data["tile_names"]
n_samples = names.shape[0]

# Extract sample IDs from tile names
SID_RX = re.compile(r"^(?P<sid>\d{1,6})")
sids = np.array([int(SID_RX.match(str(n)).group("sid")) for n in names])
unique_sids = np.array(sorted(set(sids)))

print(f"Total images: {n_samples}")
print(f"Total unique sample IDs (SIDs): {len(unique_sids)}")

# Shuffle unique sample IDs
np.random.seed(21)
np.random.shuffle(unique_sids)
sid_splits = np.array_split(unique_sids, num_folds)

# Map back to image indices
indices_list = np.empty(num_folds, dtype=object)
for i, test_sids in enumerate(sid_splits):
    indices_list[i] = np.where(np.isin(sids, test_sids))[0]
    print(f"Fold {i}: {len(test_sids)} unique SIDs mapped to {len(indices_list[i])} images.")

# Verification checks
all_sids_in_folds = np.concatenate(sid_splits)
all_indices_in_folds = np.concatenate(indices_list)

print("\n--- Split Verification ---")
print(f"All unique SIDs accounted for: {len(np.unique(all_sids_in_folds)) == len(unique_sids)}")
print(f"All images accounted for: {len(np.unique(all_indices_in_folds)) == n_samples}")

# Explicitly test for SID overlap between all pairs of folds
leakage = False
for i in range(num_folds):
    for j in range(i + 1, num_folds):
        intersection = np.intersect1d(sid_splits[i], sid_splits[j])
        if len(intersection) > 0:
            print(f"FAIL: Leakage detected between Fold {i} and Fold {j}. Overlapping SIDs: {intersection}")
            leakage = True

if not leakage:
    print("PASS: No sample ID leakage detected between folds.")
print("--------------------------\n")

parent_dir = "output_global_from_PSD_dm_500x200"
os.makedirs(parent_dir, exist_ok=True)

args.randCV_indices_path = os.path.join(parent_dir, f"random_{num_folds}_fold_indices.npy")
np.save(args.randCV_indices_path, indices_list)

print("Saved sample-level fold indices to:", args.randCV_indices_path)
print("Fold lengths (images):", [len(x) for x in indices_list])

N_runs = 5
# N_runs = num_folds for full CV

# for test_fold_index in range(N_runs):
for test_fold_index in range(4, N_runs):    # CHange back after
    args.test_fold_index = test_fold_index

    args.experiment_dir = os.path.join(
        parent_dir,
        'loss_{}'.format(args.loss_key),
        'testfold_{}'.format(args.test_fold_index)
    )
    print('******************')
    print('TEST FOLD: ', args.test_fold_index)
    print(args.experiment_dir)

    print('training...')
    run_train(args)

    print('testing...')
    run_test(args)