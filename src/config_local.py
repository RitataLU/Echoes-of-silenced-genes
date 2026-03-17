"""
Local (CPU/Mac) hyperparameter overrides.

Inherits all paths from config.py and overrides only the training-heavy
parameters so all 7 methods can run on a MacBook without a GPU.

Usage
-----
    # In local variants:
    import config_local as config
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import *   # inherit all paths, seeds, and constants

# ── GNN model (smaller for CPU) ───────────────────────────────────────────
GNN_HIDDEN_DIM  = 128
GNN_NUM_LAYERS  = 2
GNN_HEADS       = 2
GNN_PROJ_DIM    = 128
GENE_PROGRAM_K  = 32   # fewer programs → faster decode

# ── GNN training ──────────────────────────────────────────────────────────
MAX_EPOCHS      = 100
PATIENCE        = 10
LR              = 1e-3
LR_MIN          = 1e-5
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 8
GRAD_CLIP_NORM  = 1.0
LABEL_NOISE_STD = 0.0   # disable label noise for local run (faster)
EDGE_DROP_RATE  = 0.1

# ── TTA ───────────────────────────────────────────────────────────────────
TTA_SAMPLES     = 5

# ── CPA local ─────────────────────────────────────────────────────────────
CPA_LATENT_DIM  = 32
CPA_HIDDEN_DIMS = [256, 128]
CPA_EPOCHS      = 100
CPA_BATCH_SIZE  = 128

# ── MLP model ─────────────────────────────────────────────────────────────
MLP_HIDDEN_DIMS = [512, 256]
MLP_DROPOUT     = 0.3
MLP_LR          = 3e-4
MLP_EPOCHS      = 50
MLP_BATCH_SIZE  = 2048

# ── LightGBM ──────────────────────────────────────────────────────────────
LGBM_N_ESTIMATORS  = 300
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_DEPTH     = 5
LGBM_NUM_LEAVES    = 31
LGBM_SUBSAMPLE     = 0.8
LGBM_N_JOBS        = -1   # use all CPU cores
