# Advanced Methods — Local Run Guide

## Quick start (one command)

```bash
# Activate the venv first
source .venv/bin/activate

# Fast run: Ridge + CPA + visual (skip the slow GNN v2 training)
python src/run_pipeline.py --skip-gnn

# Full pipeline (includes GNN v2 — takes 2-6 hours on CPU)
python src/run_pipeline.py

# See the comparison figure at the end
python src/run_pipeline.py --skip-gnn --show-figure
```

---

## What each method does and what it produces

| Step | Script | Output file(s) | Time (Mac CPU) |
|------|--------|----------------|----------------|
| 1 | `ridge_baseline.py` | `submission_ridge.csv` | ~2 min |
| 2 | `ridge_baseline.py --pairwise` | `submission_ridge_pairwise.csv` | ~8 min |
| 3 | `train_v2.py` | `v2_fold{1-5}_best.pt`, `oof_gnn_v2.csv` | **2–6 hours** |
| 4 | `predict_v2.py` | `submission_gnn_v2.csv` | ~3 min |
| 5 | `cpa_model.py` | `cpa_best.pt`, `submission_cpa.csv` | ~20 min |
| 6 | `ensemble_v2.py` | `submission_ensemble_v2.csv` | ~1 min |
| 7 | `compare_methods.py` | `outputs/figures/method_comparison.png` | ~30 sec |

All output CSVs go to `outputs/submissions/`. Do **not** delete these existing ones:
- `submission_knn.csv`, `submission_gnn.csv`, `submission_ensemble.csv`

---

## Step-by-step commands

```bash
# Always activate the venv first
source .venv/bin/activate

# ── Step 1: Ridge baseline (fast, good first new method to try) ─────────────
python src/ridge_baseline.py

# Optional: alpha search to find the best regularisation strength
python src/ridge_baseline.py --search-alpha

# ── Step 2: Pairwise Ridge (uses pert × target gene interaction features) ────
python src/ridge_baseline.py --pairwise

# ── Step 3: Train GNN v2  ────────────────────────────────────────────────────
# Saves checkpoints to outputs/checkpoints/v2_fold{1-5}_best.pt
# Saves OOF predictions to outputs/submissions/oof_gnn_v2.csv
# WARNING: ~500 epochs × 5 folds on CPU = several hours
python src/train_v2.py

# ── Step 4: GNN v2 test predictions  ─────────────────────────────────────────
# Requires step 3 to have run (v2 checkpoints must exist)
python src/predict_v2.py

# Faster / fewer TTA samples (less accurate but quicker):
python src/predict_v2.py --tta 10

# Deterministic (no MCDropout, fastest):
python src/predict_v2.py --no-tta

# ── Step 5: CPA (Compositional Perturbation Autoencoder) ─────────────────────
# Trains AND generates submission in one command
python src/cpa_model.py

# Fewer epochs for a quicker test:
python src/cpa_model.py --epochs 100

# If you already have a checkpoint and just want predictions:
python src/cpa_model.py --predict

# ── Step 6: Ensemble v2 (blend all available submissions) ────────────────────
python src/ensemble_v2.py

# Manual weights (must sum to ~1):
python src/ensemble_v2.py --weights knn=0.15 gnn_v2=0.50 ridge=0.20 cpa=0.15

# Optimise blend weights on OOF predictions (requires oof_gnn_v2.csv):
python src/ensemble_v2.py --optimise

# ── Step 7: Compare all methods (generates the visual) ───────────────────────
python src/compare_methods.py

# Also open the figure:
python src/compare_methods.py --show
```

---

## Performance so far (before running new methods)

| Method | Validation WMAE | Type |
|--------|-----------------|------|
| Zero-predict baseline | 0.1268 | — |
| **KNN (fixed)** | **0.1232** | LOO |
| GNN v1 | 0.1239 | OOF 5-fold |
| Ensemble v1 | — | untested |

Lower is better.

---

## Understanding the new methods

### Ridge (`ridge_baseline.py`)
- Uses the same 119-dimensional gene embedding as the GNN (expression stats + coexp PCA + GO embeddings)
- Trains one Ridge regression per gene target (5127 models in total)
- **Advantage**: very fast, no GPU needed, interpretable
- **Limitation**: linear — cannot model gene–gene interactions

### Ridge Pairwise (`--pairwise`)
- Same as Ridge but also includes `pert_gene_feat × target_gene_feat` interaction terms
- Lets the model learn "which perturbed gene affects which target gene" based on feature similarity
- Slower (5127 individual models, each with 2×119 features) but often more accurate

### GNN v2 (`train_v2.py` + `predict_v2.py`)
- All improvements from GNN v1, plus:
  - **Gene programs output head**: instead of predicting 5127 raw DE values, predicts K=64 "program scores" (W @ H decomposition from training data). Dramatically reduces output dimensionality.
  - **MCDropout TTA**: keeps dropout ON at test time and averages 30 forward passes per fold. Free uncertainty-based ensemble (Gal & Ghahramani 2016).
  - **Cosine Annealing with Warm Restarts** scheduler
  - **Deeper decoder MLP** and learnable perturbation residual
- Saves checkpoints as `v2_fold{1-5}_best.pt` — different from v1 checkpoints (`fold{1-5}_best.pt`)

### CPA (`cpa_model.py`)
- Architecture: **Encoder → z_basal + δ_pert → Decoder** where δ_pert = MLP(gene_features)
- Works on individual cells (not averaged DE vectors)
- **Zero-shot capable**: the perturbation effect δ_pert is derived from the perturbed gene's features, not a lookup table — so it can generalise to unseen genes using their embedding
- Slower to train but brings a completely different inductive bias from the GNN

### Ensemble v2 (`ensemble_v2.py`)
- Blends all available submissions (KNN, GNN v1, GNN v2, Ridge, Ridge-PW, CPA) using configurable weights
- With `--optimise`: finds optimal weights via Nelder-Mead on OOF predictions

---

## Submitting to Kaggle

After running the pipeline, check the comparison figure then submit:

```bash
# View what submissions are available
ls outputs/submissions/*.csv

# Recommended submission order to try:
# 1. submission_ridge.csv        (fast to produce, good sanity check)
# 2. submission_ensemble_v2.csv  (blends all methods)
# 3. submission_gnn_v2.csv       (if GNN v2 trained)
# 4. submission_cpa.csv          (different inductive bias)
```

---

## Troubleshooting

**`No v2 checkpoints found`**: Run `train_v2.py` before `predict_v2.py`.

**`No submissions found`**: Run at least one method before `ensemble_v2.py` or `compare_methods.py`.

**GNN v2 training is too slow**: Use `--skip-gnn` with `run_pipeline.py`, or reduce epochs by editing `config.py → MAX_EPOCHS` (be aware this affects v1 too — instead copy the value after noting the original).

**CPA CUDA error**: CPA uses CPU automatically on Mac (no CUDA/MPS). It will be slow but correct.

**Out-of-memory**: Reduce `batch_size` in `cpa_model.py --batch-size 64`.

---

## Output directory reference

```
outputs/
├── checkpoints/
│   ├── fold1_best.pt … fold5_best.pt      ← GNN v1 (existing)
│   ├── v2_fold1_best.pt … v2_fold5_best.pt ← GNN v2 (new)
│   └── cpa_best.pt                         ← CPA (new)
├── submissions/
│   ├── submission_knn.csv                  ← KNN (existing, LOO WMAE 0.1232)
│   ├── submission_gnn.csv                  ← GNN v1 (existing, OOF WMAE 0.1239)
│   ├── submission_ensemble.csv             ← Ensemble v1 (existing)
│   ├── oof_gnn.csv                         ← GNN v1 OOF (existing)
│   ├── submission_ridge.csv                ← NEW
│   ├── submission_ridge_pairwise.csv       ← NEW
│   ├── oof_gnn_v2.csv                      ← NEW
│   ├── submission_gnn_v2.csv               ← NEW
│   ├── submission_cpa.csv                  ← NEW
│   └── submission_ensemble_v2.csv          ← NEW
└── figures/
    ├── training_curves.png                 ← GNN v1 curves
    ├── training_curves_v2.png              ← GNN v2 curves (new)
    └── method_comparison.png               ← 4-panel comparison (new)
```
