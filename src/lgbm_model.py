"""
LightGBM boosted-tree model for perturbation effect prediction.

Strategy
--------
Instead of fitting 5127 separate trees (too slow), we build a single model
that sees (pert_features, target_gene_features) pairs:

    X_pair = [x_pert (D), x_target (D)]  →  y = DE[pert, target]

Training: 80 perts × 5127 genes = ~410K rows
Inference: 120 perts × 5127 genes = ~615K rows

The model learns which (pert_gene, target_gene) feature combinations lead
to large DE values — a nonlinear Ridge with tree-based interactions.

5-fold CV is done on training *perturbations* (not rows) to keep the
evaluation meaningful: hold out 16 perts, train on 64.

Usage
-----
    pip install lightgbm
    python src/lgbm_model.py
    python src/lgbm_model.py --n-estimators 500 --search
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, load_pert_ids, compute_wmae, load_adata
from graph_builder import build_graph
from node_features import build_node_features
from ridge_baseline import get_gene_features


# ── Build pairwise feature matrix ─────────────────────────────────────────

def build_pairwise_X(X_perts: np.ndarray,
                     X_targets: np.ndarray) -> np.ndarray:
    """
    For each pert (n_p, D) and each target gene (n_t, D), build
    a pairwise feature matrix of shape (n_p * n_t, 2D).

    Row order: pert0-target0, pert0-target1, ..., pert0-targetN,
               pert1-target0, ...
    """
    n_p, D  = X_perts.shape
    n_t, _  = X_targets.shape

    # Repeat each pert row n_t times: (n_p * n_t, D)
    X_p = np.repeat(X_perts, n_t, axis=0)
    # Tile target rows n_p times: (n_p * n_t, D)
    X_t = np.tile(X_targets, (n_p, 1))

    return np.concatenate([X_p, X_t], axis=1).astype(np.float32)


# ── CV evaluation helper ───────────────────────────────────────────────────

def cv_evaluate(model,
                X_perts: np.ndarray,
                X_targets: np.ndarray,
                de_matrix: np.ndarray,
                weight_matrix: np.ndarray,
                n_folds: int = 5) -> float:
    """
    5-fold CV over perturbations. Returns mean OOF WMAE.
    """
    import lightgbm as lgb

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    n_t = X_targets.shape[0]
    oof_preds = np.zeros_like(de_matrix)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_perts)):
        X_tr = build_pairwise_X(X_perts[tr_idx], X_targets)   # (|tr|*n_t, 2D)
        y_tr = de_matrix[tr_idx].flatten()                     # (|tr|*n_t,)

        X_va = build_pairwise_X(X_perts[va_idx], X_targets)   # (|va|*n_t, 2D)

        m = lgb.LGBMRegressor(
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            num_leaves=model.num_leaves,
            subsample=model.subsample,
            n_jobs=model.n_jobs,
            random_state=config.SEED,
            verbose=-1,
        )
        m.fit(X_tr, y_tr)
        preds_flat = m.predict(X_va)
        oof_preds[va_idx] = preds_flat.reshape(len(va_idx), n_t)

        fold_wmae = compute_wmae(oof_preds[va_idx],
                                 de_matrix[va_idx],
                                 weight_matrix[va_idx])
        print(f"  Fold {fold+1}: WMAE = {fold_wmae:.4f}")

    oof_wmae = compute_wmae(oof_preds, de_matrix, weight_matrix)
    print(f"  OOF WMAE = {oof_wmae:.4f}  (baseline: 0.1268)")
    return oof_wmae, oof_preds


# ── Main ──────────────────────────────────────────────────────────────────

def run_lgbm(n_estimators: int = 300,
             learning_rate: float = 0.05,
             max_depth: int = 5,
             num_leaves: int = 31,
             subsample: float = 0.8,
             n_jobs: int = -1,
             cv_only: bool = False) -> str:
    """
    Train LightGBM on pairwise (pert, target_gene) features.

    Returns path to submission CSV.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    config.make_dirs()

    # Load graph + node features
    adata      = load_adata()
    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_feats = build_node_features(node_names, adata=adata)

    # Load DE + weights
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    n_t = len(gene_order)

    # Feature matrices for perturbation genes and target genes
    X_perts_train = get_gene_features(train_genes, node_names, node_feats)  # (80, D)
    X_targets     = get_gene_features(gene_order,  node_names, node_feats)  # (5127, D)

    print(f"X_perts_train: {X_perts_train.shape}")
    print(f"X_targets:     {X_targets.shape}")
    print(f"Training pairs: {X_perts_train.shape[0] * n_t:,}")

    # Normalise per-column across all gene features
    from sklearn.preprocessing import StandardScaler
    all_feats = np.vstack([X_perts_train, X_targets])
    scaler    = StandardScaler().fit(all_feats)
    X_perts_train = scaler.transform(X_perts_train)
    X_targets     = scaler.transform(X_targets)

    # Dummy model object to carry hyperparams into cv_evaluate
    class _Params:
        pass
    params = _Params()
    params.n_estimators  = n_estimators
    params.learning_rate = learning_rate
    params.max_depth     = max_depth
    params.num_leaves    = num_leaves
    params.subsample     = subsample
    params.n_jobs        = n_jobs

    print("\n5-fold CV on training perturbations:")
    oof_wmae, oof_preds = cv_evaluate(
        params, X_perts_train, X_targets, de_matrix, weight_matrix
    )

    # Save OOF
    oof_df = pd.DataFrame(oof_preds, columns=gene_order)
    oof_df.insert(0, "pert_id", train_genes)
    oof_path = os.path.join(config.SUBMISSIONS_DIR, "oof_lgbm.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    if cv_only:
        return oof_path

    # Retrain on ALL 80 training perts
    print("\nRetraining on all 80 training perturbations …")
    X_tr_full = build_pairwise_X(X_perts_train, X_targets)
    y_tr_full = de_matrix.flatten()

    final_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        n_jobs=n_jobs,
        random_state=config.SEED,
        verbose=-1,
    )
    final_model.fit(X_tr_full, y_tr_full)

    # Predict for test genes
    test_df    = load_pert_ids()
    test_genes = test_df["pert"].tolist()
    test_ids   = test_df["pert_id"].tolist()

    X_perts_test = get_gene_features(test_genes, node_names, node_feats)
    X_perts_test = scaler.transform(X_perts_test)

    X_te_full = build_pairwise_X(X_perts_test, X_targets)
    preds_flat = final_model.predict(X_te_full)
    preds = preds_flat.reshape(len(test_genes), n_t).astype(np.float32)

    # Clip to training range per gene
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)

    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_lgbm.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nLGBM submission saved: {out_path}")
    print(f"mean|pred|: {np.abs(preds).mean():.4f}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators",  type=int,   default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth",     type=int,   default=5)
    parser.add_argument("--num-leaves",    type=int,   default=31)
    parser.add_argument("--subsample",     type=float, default=0.8)
    parser.add_argument("--cv-only",       action="store_true",
                        help="Only run cross-validation, skip final submission")
    args = parser.parse_args()

    run_lgbm(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        cv_only=args.cv_only,
    )
