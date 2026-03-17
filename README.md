# Echoes of Silenced Genes — Kaggle Competition

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.4%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

Predict the transcriptional response (differential expression across **5,127 genes**) caused by CRISPRi knockdown of a single gene — without ever seeing those test genes during training.

> **Challenge**: 80 training perturbations → predict 120 unseen ones (60 val + 60 test). Zero overlap.

---

## Competition Overview

| | |
|---|---|
| **Task** | Predict DE vector for each test gene knockdown |
| **Ground truth** | `DE[pert, gene] = mean(perturbed cells) − mean(non-targeting cells)` |
| **Evaluation** | Weighted MAE (weights vary per perturbation AND per gene) |
| **Baseline WMAE** | 0.1268 (mean-DE vector) · Zero predictor: 0.1413 |
| **Dataset** | 17,882 cells × 19,226 genes (scRNA-seq, h5ad format) |

---

## Solution Overview

Three-phase approach, from simple to complex:

```
[Phase 1]  STRING KNN Baseline                    →  submission_knn.csv
[Phase 2]  GEARS-style GAT (5-fold CV)            →  submission_gnn.csv
[Phase 3]  Ridge · GNN v2 · CPA · GNN v3 · Blend →  submission_gnn_v3_local.csv  ← best
```

### Results

| Method | Val WMAE | Notes |
|--------|----------|-------|
| Zero predictor | 0.1413 | predict all zeros |
| Mean-DE baseline | 0.1268 | competition baseline |
| **KNN (STRING)** | **0.1232** | Phase 1, LOO eval |
| GNN v1 (GAT) | 0.1239 | Phase 2, 5-fold OOF |
| Ridge (linear) | ~0.120 | Phase 3 |
| GNN v2 + MCDropout | ~0.115 | Phase 3 |
| CPA | ~0.118 | Phase 3 |
| Ensemble v2 | ~0.110 | Phase 3, blended |
| **GNN v3 (TransformerConv + TF edges)** | **best** | Phase 3 ← final submission |

---

## Architecture

### Phase 1 — STRING KNN
For each unseen test gene, find the *k*=5 most functionally similar training genes using STRING PPI scores (or Pearson co-expression as fallback), then predict as a softmax-weighted average of their DE vectors.

### Phase 2 — GEARS-style 2-Stage GAT

```
Gene interaction graph:
  Nodes = 5,127 output genes ∪ 80 train perts ∪ 120 test perts
  Edges = STRING PPI (score ≥ 700) + co-expression (|r| ≥ 0.4, top-20/gene)

Node features (119D):
  [5D expression stats] + [50D co-expression PCA] + [64D GO term SVD]
   ↓ MLP projection → 128D

Stage 1 — Context Encoder (perturbation-agnostic, cached per epoch):
  128D → GAT(heads=2) → GAT(heads=2) → h_base (128D per gene)

Stage 2 — Perturbation Propagator (one forward pass per perturbation):
  pert_signal = MLP(h_base[perturbed_gene_idx])
  input = concat(h_base, pert_signal) → 128D
  → GAT + soft residual (h + 0.1·h_base)  ×2
  → linear decoder on 5,127 output nodes
  → DE predictions (5,127 values)
```

**Zero-shot key**: perturbation signal is derived from the *gene's own graph embedding*, not a learnable lookup table → generalises to any gene in the graph.

**Training**: 5-fold CV · DropEdge 10% · label noise σ=0.001 · Huber→WMAE loss · cosine LR annealing · early stopping

### Phase 3 — Additional Methods

| Model | Description |
|-------|-------------|
| **Ridge** | 5,127 Ridge regressors on 119D gene embeddings. Fast, interpretable. |
| **Ridge Pairwise** | Adds `pert_feat × target_feat` interaction terms. |
| **GNN v2** | GNN v1 + gene-program output head (K=64) + MCDropout TTA (30 passes/fold) + warm restarts |
| **CPA** | Compositional Perturbation Autoencoder: `Encoder → z_basal + δ_pert → Decoder`. Cell-level, different inductive bias. |
| **GNN v3** ← best | TransformerConv (dot-product attention) + directed TF regulatory edges (TRRUST) + edge-type encoding + optional global context layer (VirtualNode-style). Improves on v2 by letting the model distinguish PPI, TF-activating, and TF-repressing signals. |
| **Ensemble v2** | Blend all submissions; `--optimise` finds weights via Nelder-Mead on OOF predictions. |

---

## Project Structure

```
├── src/
│   ├── config.py              # paths + hyperparameters
│   ├── data_utils.py          # data loading, WMAE computation
│   ├── graph_builder.py       # STRING download + co-expression + build_graph
│   ├── node_features.py       # 119D node feature matrix (cached)
│   ├── knn_baseline.py        # Phase 1: STRING KNN submission
│   ├── gnn_model.py           # Phase 2: 2-stage GAT model
│   ├── dataset.py             # PerturbationDataset (static graph)
│   ├── train.py               # 5-fold CV training (GNN v1)
│   ├── predict.py             # GNN v1 submission
│   ├── ensemble.py            # Phase 2 ensemble (GNN + KNN)
│   ├── ridge_baseline.py      # Phase 3: Ridge / Ridge-Pairwise
│   ├── train_v2.py            # Phase 3: GNN v2 training
│   ├── predict_v2.py          # Phase 3: GNN v2 submission (+ MCDropout TTA)
│   ├── cpa_model.py           # Phase 3: CPA model
│   ├── graph_builder_v3.py    # Phase 3: graph with directed TF edges (TRRUST)
│   ├── gnn_model_v3.py        # Phase 3: GNN v3 (TransformerConv + edge types)
│   ├── train_v3_local.py      # Phase 3: GNN v3 training
│   ├── predict_v3_local.py    # Phase 3: GNN v3 submission ← final
│   ├── ensemble_v2.py         # Phase 3: blend all submissions
│   ├── compare_methods.py     # generate method comparison figure
│   └── run_pipeline.py        # one-command pipeline runner
├── notebooks/
│   ├── 01_eda.ipynb           # 9 EDA visualizations
│   ├── 02_model_analysis.ipynb
│   ├── 03_gnn_v3_analysis.ipynb
│   └── 04_submission_viz.ipynb
├── outputs/
│   ├── checkpoints/           # model weights per fold (git-ignored)
│   ├── submissions/           # CSV files (git-ignored)
│   ├── figures/               # saved plots
│   └── cache/                 # precomputed features (git-ignored)
├── external/                  # downloaded external data (git-ignored)
│   ├── string_interactions.tsv
│   ├── go-basic.obo
│   └── goa_human.gaf.gz
├── data/                      # Kaggle competition data (git-ignored)
├── requirements.txt
├── RUN_GUIDE.md               # detailed step-by-step run guide
└── PROJECT_REPORT.md          # full project report (Traditional Chinese)
```

> `data/` and large generated files are excluded from the repo. See [Data Setup](#data-setup) below.

---

## Setup

```bash
# 1. Clone and create virtual environment
git clone https://github.com/<you>/echoes-of-silenced-genes.git
cd echoes-of-silenced-genes
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch first
pip install torch>=2.1.0

# 3. Install PyTorch Geometric (Mac CPU/MPS)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Verify Apple Silicon GPU
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### Data Setup

Competition data must be downloaded separately from Kaggle (terms prohibit redistribution):

```bash
# Place competition files under:
data/echoes-of-silenced-genes/
├── training_cells.h5ad
├── training_data_ground_truth_table.csv
├── training_data_means.csv
├── pert_ids_all.csv
└── sample_submission.csv
```

External biological databases are downloaded automatically by the pipeline scripts (see below).

---

## How to Run

### Quick start (Phase 3 pipeline)

```bash
source .venv/bin/activate

# Fast: Ridge + CPA + comparison figure (skips slow GNN v2)
python src/run_pipeline.py --skip-gnn

# Full pipeline including GNN v2 (2–6 hours on CPU)
python src/run_pipeline.py
```

### Step by step

```bash
# ── Step 1: Download external data (one-time) ────────────────────────────────
python src/graph_builder.py --download-string   # STRING PPI
python src/graph_builder.py --download-go       # Gene Ontology

# ── Step 2: KNN baseline (no training needed) ────────────────────────────────
python src/knn_baseline.py
# → outputs/submissions/submission_knn.csv

# ── Step 3: Build node feature cache ─────────────────────────────────────────
python src/node_features.py
# → outputs/cache/node_features.npy

# ── Step 4: GNN v1 (5-fold CV) ───────────────────────────────────────────────
python src/train.py
python src/predict.py
# → outputs/submissions/submission_gnn.csv

# ── Step 5: Ridge ─────────────────────────────────────────────────────────────
python src/ridge_baseline.py           # standard Ridge (~2 min)
python src/ridge_baseline.py --pairwise  # with interaction features (~8 min)

# ── Step 6: GNN v2 ───────────────────────────────────────────────────────────
python src/train_v2.py                 # ~2–6 hours on CPU
python src/predict_v2.py              # MCDropout TTA (30 passes/fold)

# ── Step 7: CPA ──────────────────────────────────────────────────────────────
python src/cpa_model.py               # trains + generates submission (~20 min)

# ── Step 8: GNN v3 (best model) ──────────────────────────────────────────────
python src/train_v3_local.py          # TransformerConv + TF edges (~2–6 hours)
python src/predict_v3_local.py        # → outputs/submissions/submission_gnn_v3_local.csv

# ── Step 9: Ensemble ─────────────────────────────────────────────────────────
python src/ensemble_v2.py             # blends all available submissions
python src/ensemble_v2.py --optimise  # find optimal weights on OOF predictions

# ── Step 10: Compare ────────────────────────────────────────────────────────
python src/compare_methods.py --show
```

See [RUN_GUIDE.md](RUN_GUIDE.md) for full details, timing estimates, and troubleshooting.

---

## Key Design Decisions

1. **Zero-shot generalisation**: Perturbation signal = `MLP(h_base[pert_gene_idx])`, not a lookup table. Any gene with graph embeddings (even unseen ones) can be perturbed at inference time.

2. **Multi-source biological priors**: STRING PPI (protein interactions) + scRNA-seq co-expression + Gene Ontology term embeddings, all fused into 119D node features.

3. **Loss = evaluation metric**: Training directly optimises Weighted MAE using the competition's exact weight matrix (per-perturbation × per-gene).

4. **Regularisation for small datasets**: Only 80 training examples → DropEdge (10%) + label noise (σ=0.001) + soft residual connections (`h + 0.1·h_base`) to prevent collapse.

5. **MCDropout TTA**: GNN v2 keeps dropout active at test time and averages 30 forward passes per fold — a cheap uncertainty-aware ensemble (Gal & Ghahramani, 2016).

---

## Acknowledgements

- [GEARS](https://github.com/snap-stanford/GEARS) — inspiration for the 2-stage GNN design
- [STRING database](https://string-db.org/) — protein–protein interaction network
- [Gene Ontology](http://geneontology.org/) — functional gene annotations
- [PyTorch Geometric](https://pyg.org/) — graph neural network framework
