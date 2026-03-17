"""
Central configuration: all paths and hyperparameters live here.
Import this in every other module instead of hardcoding paths.
"""
from __future__ import annotations
import os

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(ROOT_DIR, "data", "echoes-of-silenced-genes")
EXTERNAL_DIR = os.path.join(ROOT_DIR, "external")
OUTPUT_DIR   = os.path.join(ROOT_DIR, "outputs")

CHECKPOINTS_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSIONS_DIR  = os.path.join(OUTPUT_DIR, "submissions")
FIGURES_DIR      = os.path.join(OUTPUT_DIR, "figures")
CACHE_DIR        = os.path.join(OUTPUT_DIR, "cache")   # for intermediate numpy arrays

# Data files
H5AD_PATH         = os.path.join(DATA_DIR, "training_cells.h5ad")
MEANS_PATH        = os.path.join(DATA_DIR, "training_data_means.csv")
GT_PATH           = os.path.join(DATA_DIR, "training_data_ground_truth_table.csv")
PERT_IDS_ALL_PATH = os.path.join(DATA_DIR, "pert_ids_all.csv")
PERT_IDS_VAL_PATH = os.path.join(DATA_DIR, "pert_ids_val.csv")
SAMPLE_SUB_PATH   = os.path.join(DATA_DIR, "sample_submission.csv")

# External files
STRING_TSV_PATH   = os.path.join(EXTERNAL_DIR, "string_interactions.tsv")
GO_OBO_PATH       = os.path.join(EXTERNAL_DIR, "go-basic.obo")
GO_GAF_PATH       = os.path.join(EXTERNAL_DIR, "goa_human.gaf.gz")

# Cached arrays
NODE_FEATURES_PATH = os.path.join(CACHE_DIR, "node_features.npy")
NODE_NAMES_PATH    = os.path.join(CACHE_DIR, "node_names.txt")
COEXP_EMBED_PATH   = os.path.join(CACHE_DIR, "coexp_embed.npy")
GO_EMBED_PATH      = os.path.join(CACHE_DIR, "go_embed.npy")
ESM2_EMBED_PATH    = os.path.join(CACHE_DIR, "esm2_embed.npy")
ESM2_GENE_NAMES    = os.path.join(CACHE_DIR, "esm2_gene_names.txt")
GENE_PROGRAM_H     = os.path.join(CACHE_DIR, "gene_program_H.npy")   # (K, 5127)
GENE_PROGRAM_NAMES = os.path.join(CACHE_DIR, "gene_program_gene_names.txt")

# ── STRING settings ────────────────────────────────────────────────────────
STRING_SPECIES          = 9606    # Homo sapiens
STRING_SCORE_KNN        = 400     # lower threshold for KNN (more connections)
STRING_SCORE_GNN        = 700     # high-confidence threshold for GNN edges
STRING_VERSION          = "12.0"
STRING_COEXP_THRESHOLD  = 0.4     # Pearson |r| threshold for co-expression edges
STRING_COEXP_TOP_K      = 20      # max co-expression edges per gene

# ── Node feature dimensions ────────────────────────────────────────────────
EXPR_STATS_DIM   = 5     # mean, log1p_mean, var, dispersion, detect_rate
COEXP_PCA_DIM    = 50    # TruncatedSVD on control expression matrix
GO_EMBED_DIM     = 64    # TruncatedSVD on gene × GO-term binary matrix
NODE_FEAT_DIM    = EXPR_STATS_DIM + COEXP_PCA_DIM + GO_EMBED_DIM  # = 119 → projected to 128

# ── GNN hyperparameters ────────────────────────────────────────────────────
GNN_HIDDEN_DIM   = 256    # GPU: 256 (was 128 on local Mac)
GNN_NUM_LAYERS   = 3      # GPU: 3   (was 2)
GNN_DROPOUT      = 0.2
GNN_HEADS        = 4      # GPU: 4   (was 2)
GNN_PROJ_DIM     = 256
EDGE_DROP_RATE   = 0.10

# ── Gene Programs (SVD output decomposition) ───────────────────────────────
USE_GENE_PROGRAMS   = True   # decompose output space via SVD
GENE_PROGRAM_K      = 64     # number of latent gene programs
GP_CACHE_PATH       = None   # set dynamically in gene_programs.py

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE       = 16
LR               = 3e-4      # lower LR for larger model
WEIGHT_DECAY     = 1e-4
MAX_EPOCHS       = 500        # GPU: 500
PATIENCE         = 20         # 20 validation checks = 200 epochs patience
LR_MIN           = 1e-6
LABEL_NOISE_STD  = 0.001
GRAD_CLIP_NORM   = 1.0
N_FOLDS          = 5
SEED             = 42

# ── Test-Time Augmentation (MCDropout) ────────────────────────────────────
TTA_SAMPLES      = 30         # number of forward passes with dropout ON

# ── KNN baseline ──────────────────────────────────────────────────────────
KNN_K            = 5

# ── Ensemble ──────────────────────────────────────────────────────────────
ENSEMBLE_ALPHA   = 0.7    # GNN weight; (1 - alpha) = KNN weight


def make_dirs():
    """Create all output directories if they don't exist."""
    for d in [EXTERNAL_DIR, CHECKPOINTS_DIR, SUBMISSIONS_DIR, FIGURES_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    make_dirs()
    print("All directories created.")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
