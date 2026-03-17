"""
Phase 3: Ensemble GNN and KNN predictions.

Blends submission_gnn.csv and submission_knn.csv with a configurable alpha:
    final = alpha * gnn + (1 - alpha) * knn

Alpha is optimised on the val split (60 perturbations whose pert class = 'val')
if ground truth is available (training genes do not overlap with val, so we
use OOF predictions to proxy this).

Usage:
    python src/ensemble.py
    python src/ensemble.py --alpha 0.7   # fixed alpha, skip optimisation
    python src/ensemble.py --optimise-alpha   # grid search alpha on OOF
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, load_pert_ids, compute_wmae


def load_submission(name: str) -> pd.DataFrame:
    path = os.path.join(config.SUBMISSIONS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Submission not found: {path}. Run the corresponding script first.")
    return pd.read_csv(path)


def blend(gnn_df: pd.DataFrame,
          knn_df: pd.DataFrame,
          alpha: float) -> pd.DataFrame:
    """
    Blend GNN and KNN predictions.
    Both DataFrames must have pert_id as first column and identical gene columns.

    Returns a blended DataFrame in the same format.
    """
    assert list(gnn_df.columns) == list(knn_df.columns), "Column mismatch between GNN and KNN submissions"

    pert_ids = gnn_df["pert_id"].values
    gene_cols = [c for c in gnn_df.columns if c != "pert_id"]

    gnn_vals = gnn_df[gene_cols].values.astype(np.float32)
    knn_vals = knn_df[gene_cols].values.astype(np.float32)

    # Align by pert_id order (knn might be in different order)
    knn_order = dict(zip(knn_df["pert_id"], range(len(knn_df))))
    knn_reordered = np.array([knn_vals[knn_order[pid]] for pid in pert_ids])

    blended = alpha * gnn_vals + (1 - alpha) * knn_reordered

    result = pd.DataFrame(blended, columns=gene_cols)
    result.insert(0, "pert_id", pert_ids)
    return result


def optimise_alpha(gnn_df: pd.DataFrame,
                   knn_df: pd.DataFrame,
                   oof_gnn_df: pd.DataFrame) -> float:
    """
    Grid search alpha on OOF training predictions vs ground truth.
    Uses OOF GNN predictions since we cannot evaluate on the real test set.

    Returns best alpha.
    """
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    # Load KNN predictions for training genes (if available)
    knn_train_path = os.path.join(config.SUBMISSIONS_DIR, "submission_knn.csv")
    knn_all = pd.read_csv(knn_train_path)

    # Filter to training genes that appear in the KNN submission
    # (KNN is for test genes only; for training we use OOF GNN vs mean DE)
    # → Fall back to a fixed heuristic: blend OOF GNN with mean-DE baseline
    mean_de = de_matrix.mean(axis=0, keepdims=True)
    mean_de_repeated = np.tile(mean_de, (de_matrix.shape[0], 1))

    # OOF GNN preds
    oof_order = {g: i for i, g in enumerate(oof_gnn_df["pert_id"].values)}
    oof_preds = np.array([
        oof_gnn_df.iloc[oof_order[g]][[c for c in oof_gnn_df.columns if c != "pert_id"]].values
        for g in train_genes
    ], dtype=np.float32)

    best_alpha = 1.0
    best_wmae  = compute_wmae(oof_preds, de_matrix, weight_matrix)
    print(f"Baseline (GNN OOF only): WMAE = {best_wmae:.4f}")

    for alpha in np.arange(0.0, 1.05, 0.05):
        blended = alpha * oof_preds + (1 - alpha) * mean_de_repeated
        wmae = compute_wmae(blended, de_matrix, weight_matrix)
        print(f"  alpha={alpha:.2f}: WMAE = {wmae:.4f}")
        if wmae < best_wmae:
            best_wmae = wmae
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha:.2f} → WMAE = {best_wmae:.4f}")
    return best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None,
                        help=f"Fixed blend weight for GNN (default: {config.ENSEMBLE_ALPHA}). "
                             "1.0 = GNN only, 0.0 = KNN only.")
    parser.add_argument("--optimise-alpha", action="store_true",
                        help="Grid search alpha on OOF GNN predictions vs GT")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv")
    args = parser.parse_args()

    config.make_dirs()

    print("Loading GNN submission …")
    gnn_df = load_submission("submission_gnn.csv")

    print("Loading KNN submission …")
    knn_df = load_submission("submission_knn.csv")

    alpha = args.alpha

    if args.optimise_alpha:
        oof_path = os.path.join(config.SUBMISSIONS_DIR, "oof_gnn.csv")
        if os.path.exists(oof_path):
            oof_gnn_df = pd.read_csv(oof_path)
            alpha = optimise_alpha(gnn_df, knn_df, oof_gnn_df)
        else:
            print("OOF predictions not found — using default alpha.")

    if alpha is None:
        alpha = config.ENSEMBLE_ALPHA

    print(f"\nBlending GNN × {alpha:.2f} + KNN × {1-alpha:.2f} …")
    blended_df = blend(gnn_df, knn_df, alpha=alpha)

    out_path = os.path.join(config.SUBMISSIONS_DIR, args.output)
    blended_df.to_csv(out_path, index=False)
    print(f"Ensemble submission saved to: {out_path}")

    # Quick stats
    gene_cols = [c for c in blended_df.columns if c != "pert_id"]
    vals = blended_df[gene_cols].values
    print(f"\nEnsemble stats:")
    print(f"  Shape: {vals.shape}")
    print(f"  mean |pred|: {np.abs(vals).mean():.4f}")
    print(f"  max  |pred|: {np.abs(vals).max():.4f}")


if __name__ == "__main__":
    main()
