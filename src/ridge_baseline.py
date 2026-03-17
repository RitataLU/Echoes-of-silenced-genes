"""
Ridge Regression baseline — per-gene regression on perturbation embeddings.

Why this can beat GNN with only 80 training samples
----------------------------------------------------
The GNN tries to jointly predict 5127 output values through a complex
message-passing architecture. With only 80 training perturbations, this
often leads to underfitting / poor generalisation.

Ridge regression takes a completely different view:

    For EACH target gene j independently:
        Train a linear model: f_j(x_pert) → DE[pert, j]
        where x_pert is the feature vector of the PERTURBED gene (not the target gene)

    With:
        X_train : (80, D)   — feature vectors of 80 training perturbation genes
        y_train : (80,)     — DE values for gene j across all 80 perturbations
        → fit Ridge(alpha)

    At test time:
        X_test  : (120, D)  — feature vectors of 120 test perturbation genes
        → predict (120,) DE values for gene j

This is 5127 separate regressions, each with 80 samples and D~119 features.
With ridge regularisation, this is well-conditioned and often very competitive.

Key insight: the INPUT features represent the perturbed gene's biology.
If two genes (train and test) have similar STRING/GO/expression profiles,
their perturbation effects on gene j will also be similar.

Stacking trick
--------------
Instead of fitting 5127 separate models, we use a single Ridge with
multi-output support: Ridge(X_train, DE_train) — much faster.

Usage
-----
    python src/ridge_baseline.py
    python src/ridge_baseline.py --alpha 10.0
    python src/ridge_baseline.py --search-alpha   # grid search over alpha
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, load_pert_ids, compute_wmae
from graph_builder import build_graph
from node_features import build_node_features
from data_utils import load_adata


# ── Feature extraction ────────────────────────────────────────────────────

def get_gene_features(gene_list: list[str],
                       node_names: list[str],
                       node_features: np.ndarray) -> np.ndarray:
    """
    Look up pre-computed node features for a list of gene names.

    Returns (n_genes, feat_dim) float32. Genes not in the graph get zeros.
    """
    gene_to_idx = {g: i for i, g in enumerate(node_names)}
    feat_dim = node_features.shape[1]
    out = np.zeros((len(gene_list), feat_dim), dtype=np.float32)
    for i, g in enumerate(gene_list):
        if g in gene_to_idx:
            out[i] = node_features[gene_to_idx[g]]
    return out


# ── Alpha search ──────────────────────────────────────────────────────────

def search_alpha(X_train: np.ndarray,
                 de_matrix: np.ndarray,
                 weight_matrix: np.ndarray,
                 alphas: list[float] | None = None,
                 n_folds: int = 5) -> float:
    """
    Grid search over Ridge alpha using leave-one-out style CV on training genes.

    Returns best alpha.
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    scaler = StandardScaler()

    best_alpha = alphas[0]
    best_wmae  = float("inf")

    print("Alpha search (Ridge, 5-fold CV on training genes):")
    for alpha in alphas:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr = scaler.fit_transform(X_train[train_idx])
            X_va = scaler.transform(X_train[val_idx])

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_tr, de_matrix[train_idx])
            preds = model.predict(X_va)

            wmae = compute_wmae(preds, de_matrix[val_idx], weight_matrix[val_idx])
            fold_scores.append(wmae)

        mean_wmae = np.mean(fold_scores)
        print(f"  alpha={alpha:8.2f}  →  CV WMAE = {mean_wmae:.4f}")

        if mean_wmae < best_wmae:
            best_wmae  = mean_wmae
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}  →  WMAE = {best_wmae:.4f}")
    return best_alpha


# ── Main prediction ───────────────────────────────────────────────────────

def run_ridge(alpha: float = 100.0,
              use_elastic: bool = False,
              search: bool = False) -> str:
    """
    Train Ridge on training gene embeddings → predict for test genes.

    Returns path to submission CSV.
    """
    config.make_dirs()

    # Load graph + node features
    adata = load_adata()
    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_features = build_node_features(node_names, adata=adata)

    # Load DE + weights
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    # Feature matrices
    X_train = get_gene_features(train_genes, node_names, node_features)  # (80, D)
    print(f"X_train: {X_train.shape}  (80 training pert genes × {X_train.shape[1]} features)")

    # Optional: also concatenate mean expression of the pert gene
    # (adds a simple "how expressed is this gene?" signal)
    # This is already in the node features (expr_stats component), so skip.

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Alpha search
    if search:
        alpha = search_alpha(X_train_scaled, de_matrix, weight_matrix)
    else:
        # Quick LOO evaluation at the chosen alpha
        kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
        fold_scores = []
        for tr_idx, va_idx in kf.split(X_train_scaled):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(X_train_scaled[tr_idx], de_matrix[tr_idx])
            preds = m.predict(X_train_scaled[va_idx])
            fold_scores.append(
                compute_wmae(preds, de_matrix[va_idx], weight_matrix[va_idx])
            )
        cv_wmae = np.mean(fold_scores)
        print(f"5-fold CV WMAE (alpha={alpha}): {cv_wmae:.4f}  (baseline: 0.1268)")

    # Train on ALL 80 training genes
    if use_elastic:
        # ElasticNet: Ridge + Lasso, forces sparse coefficients
        # Useful when many features are irrelevant
        final_model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)
    else:
        final_model = Ridge(alpha=alpha, fit_intercept=True)

    final_model.fit(X_train_scaled, de_matrix)   # (80, D) → (80, 5127)

    # Predict for test genes
    test_df    = load_pert_ids()
    test_genes = test_df["pert"].tolist()
    test_ids   = test_df["pert_id"].tolist()

    X_test        = get_gene_features(test_genes, node_names, node_features)
    X_test_scaled = scaler.transform(X_test)
    preds         = final_model.predict(X_test_scaled)   # (120, 5127)

    # Clip to training range per gene
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max).astype(np.float32)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)

    name = "submission_ridge_elastic.csv" if use_elastic else "submission_ridge.csv"
    out_path = os.path.join(config.SUBMISSIONS_DIR, name)
    sub.to_csv(out_path, index=False)
    print(f"\nRidge submission saved: {out_path}")
    print(f"mean|pred|: {np.abs(preds).mean():.4f}")
    return out_path


# ── Augmented Ridge: perturbation gene × target gene interaction ──────────

def run_ridge_pairwise(alpha: float = 10.0) -> str:
    """
    Enhanced Ridge that uses pairwise features:
        x_input = [x_pert, x_target_gene, x_pert * x_target_gene]

    For each (perturbation gene, target gene) pair, the input includes
    both the perturbed gene's features AND the target gene's features.
    This lets the model learn which perturbed gene affects which target gene
    based on their INTERACTION in feature space.

    This is more powerful than standard Ridge but still linear.
    Trade-off: 5127 separate models (one per target gene), each with 3D features.
    """
    config.make_dirs()

    adata = load_adata()
    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_features = build_node_features(node_names, adata=adata)

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    test_df    = load_pert_ids()
    test_genes = test_df["pert"].tolist()
    test_ids   = test_df["pert_id"].tolist()

    X_perts_train = get_gene_features(train_genes, node_names, node_features)  # (80, D)
    X_perts_test  = get_gene_features(test_genes,  node_names, node_features)  # (120, D)
    D = X_perts_train.shape[1]

    gene_to_idx = {g: i for i, g in enumerate(node_names)}
    n_output    = len(gene_order)
    preds       = np.zeros((120, n_output), dtype=np.float32)

    print(f"Pairwise Ridge: predicting {n_output} target genes …")

    scaler_pert = StandardScaler().fit(X_perts_train)
    Xp_tr = scaler_pert.transform(X_perts_train)   # (80, D)
    Xp_te = scaler_pert.transform(X_perts_test)    # (120, D)

    LOG_EVERY = 500
    for j, target_gene in enumerate(gene_order):
        t_idx = gene_to_idx.get(target_gene)
        if t_idx is not None:
            x_t = node_features[t_idx]                      # (D,)
            x_t = x_t / (np.linalg.norm(x_t) + 1e-8)       # unit norm

            # Interaction: pert_feat ⊙ target_feat
            inter_tr = Xp_tr * x_t[None, :]                 # (80, D)
            inter_te = Xp_te * x_t[None, :]                 # (120, D)

            X_tr_j = np.concatenate([Xp_tr, inter_tr], axis=1)  # (80, 2D)
            X_te_j = np.concatenate([Xp_te, inter_te], axis=1)  # (120, 2D)
        else:
            X_tr_j = Xp_tr
            X_te_j = Xp_te

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_tr_j, de_matrix[:, j])
        preds[:, j] = model.predict(X_te_j)

        if (j + 1) % LOG_EVERY == 0:
            print(f"  {j+1}/{n_output} genes processed")

    # Clip
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)

    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_ridge_pairwise.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nPairwise Ridge submission saved: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",       type=float, default=100.0)
    parser.add_argument("--search-alpha",action="store_true")
    parser.add_argument("--elastic",     action="store_true")
    parser.add_argument("--pairwise",    action="store_true",
                        help="Use pairwise pert×target features (slower but better)")
    args = parser.parse_args()

    if args.pairwise:
        run_ridge_pairwise(alpha=args.alpha)
    else:
        run_ridge(alpha=args.alpha, use_elastic=args.elastic, search=args.search_alpha)
