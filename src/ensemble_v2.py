"""
Ensemble v2 — blend all available submissions with OOF-optimised weights.

Available submissions to blend:
    submission_knn.csv           — STRING KNN baseline
    submission_gnn_v2.csv        — GNN v2 (gene programs + MCDropout)
    submission_ridge.csv         — Ridge regression on gene embeddings
    submission_ridge_pairwise.csv— Ridge with pert × target interaction features
    submission_cpa.csv           — CPA (cell-level autoencoder)

Strategy
--------
1. For each method that has OOF predictions on training genes, compute the
   OOF WMAE.
2. Use Nelder-Mead / grid search to find per-method weights that minimise
   a weighted combination of OOF WMAEs.
3. Final submission = weighted average of all methods.

Usage
-----
    # Auto-blend all available submissions (equal weights if no OOF):
    python src/ensemble_v2.py

    # Optimise weights using OOF predictions:
    python src/ensemble_v2.py --optimise

    # Fix specific weights:
    python src/ensemble_v2.py --weights knn=0.1 gnn=0.5 ridge=0.2 cpa=0.2
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, compute_wmae


# ── Load helpers ──────────────────────────────────────────────────────────

def load_sub(name: str) -> pd.DataFrame | None:
    path = os.path.join(config.SUBMISSIONS_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def sub_to_array(df: pd.DataFrame, gene_order: list[str]) -> np.ndarray:
    """Extract prediction matrix in gene_order column order."""
    return df[[c for c in gene_order if c in df.columns]].values.astype(np.float32)


def oof_to_array(df: pd.DataFrame,
                 train_genes: list[str],
                 gene_order: list[str]) -> np.ndarray | None:
    if df is None:
        return None
    gene_cols = [c for c in gene_order if c in df.columns]
    if "pert_id" not in df.columns:
        return None
    order = {g: i for i, g in enumerate(df["pert_id"].tolist())}
    out = np.zeros((len(train_genes), len(gene_cols)), dtype=np.float32)
    for i, g in enumerate(train_genes):
        if g in order:
            out[i] = df.iloc[order[g]][gene_cols].values
    return out


# ── Weight optimisation ───────────────────────────────────────────────────

def optimise_weights(oof_preds: dict[str, np.ndarray],
                     de_matrix: np.ndarray,
                     weight_matrix: np.ndarray) -> dict[str, float]:
    """
    Find blend weights that minimise OOF WMAE using Nelder-Mead.

    oof_preds: {method_name: (n_train, n_genes)} arrays
    Returns  : {method_name: weight} summing to 1.0
    """
    names  = list(oof_preds.keys())
    arrays = [oof_preds[n] for n in names]
    n      = len(names)

    def wmae_blend(raw_weights):
        # Softmax so weights sum to 1 and stay positive
        w = np.exp(raw_weights)
        w = w / w.sum()
        blended = sum(w[i] * arrays[i] for i in range(n))
        return compute_wmae(blended, de_matrix, weight_matrix)

    # Initialise with equal weights
    x0 = np.zeros(n)
    result = minimize(wmae_blend, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})

    raw = result.x
    final_w = np.exp(raw) / np.exp(raw).sum()
    weights = {names[i]: float(final_w[i]) for i in range(n)}

    print(f"\nOptimised blend weights (OOF WMAE: {result.fun:.4f}):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}  {w:.3f}")
    return weights


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimise", action="store_true",
                        help="Optimise blend weights on OOF predictions")
    parser.add_argument("--weights",  nargs="*", default=None,
                        help="Manual weights e.g. knn=0.1 gnn=0.5 ridge=0.2 cpa=0.2")
    parser.add_argument("--output",   type=str, default="submission_ensemble_v2.csv")
    args = parser.parse_args()

    config.make_dirs()
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    # ── Load all available submissions ────────────────────────────────
    subs = {
        "knn":             load_sub("submission_knn.csv"),
        "gnn_v2":          load_sub("submission_gnn_v2.csv"),
        "gnn_v1":          load_sub("submission_gnn.csv"),
        "ridge":           load_sub("submission_ridge.csv"),
        "ridge_pairwise":  load_sub("submission_ridge_pairwise.csv"),
        "cpa":             load_sub("submission_cpa.csv"),
    }
    available = {k: v for k, v in subs.items() if v is not None}
    print(f"Available submissions: {list(available.keys())}")

    if not available:
        raise FileNotFoundError("No submissions found. Run at least one method first.")

    # ── OOF arrays (for weight optimisation) ──────────────────────────
    oof_files = {
        "gnn_v2":  load_sub("oof_gnn_v2.csv"),
        "gnn_v1":  load_sub("oof_gnn.csv"),
    }
    oof_arrays = {
        k: oof_to_array(v, train_genes, gene_order)
        for k, v in oof_files.items()
        if v is not None
    }

    # Baseline OOF for comparison
    mean_de = de_matrix.mean(axis=0, keepdims=True)
    for name, oof in oof_arrays.items():
        wmae = compute_wmae(oof, de_matrix, weight_matrix)
        print(f"  OOF WMAE [{name}]: {wmae:.4f}  (baseline: 0.1268)")

    # ── Determine blend weights ────────────────────────────────────────
    if args.weights:
        # Manual: parse "knn=0.1 gnn=0.5 ..."
        weights = {}
        for item in args.weights:
            k, v = item.split("=")
            weights[k] = float(v)
        # Normalise
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    elif args.optimise and len(oof_arrays) >= 2:
        weights = optimise_weights(oof_arrays, de_matrix, weight_matrix)
        # Fill in equal weights for methods without OOF
        n_no_oof = len(available) - len(oof_arrays)
        if n_no_oof > 0:
            residual = 1.0 - sum(weights.values())
            share    = residual / n_no_oof
            for name in available:
                if name not in weights:
                    weights[name] = share

    else:
        # Equal weights across all available submissions
        n = len(available)
        weights = {k: 1.0 / n for k in available}
        print(f"\nUsing equal weights (1/{n} each): {weights}")

    # ── Blend ──────────────────────────────────────────────────────────
    # Align all predictions by pert_id order from the first available sub
    ref_df      = next(iter(available.values()))
    pert_ids    = ref_df["pert_id"].tolist()

    blended = np.zeros((len(pert_ids), len(gene_order)), dtype=np.float64)
    total_w = 0.0
    for name, w in weights.items():
        if name not in available:
            continue
        df   = available[name]
        arr  = sub_to_array(df.set_index("pert_id").reindex(pert_ids).reset_index(),
                             gene_order)
        blended += w * arr
        total_w += w

    blended = (blended / total_w).astype(np.float32)

    # ── Save ───────────────────────────────────────────────────────────
    out_df = pd.DataFrame(blended, columns=gene_order)
    out_df.insert(0, "pert_id", pert_ids)
    out_path = os.path.join(config.SUBMISSIONS_DIR, args.output)
    out_df.to_csv(out_path, index=False)

    print(f"\nEnsemble v2 saved: {out_path}")
    print(f"  Shape: {blended.shape}")
    print(f"  mean|pred|: {np.abs(blended).mean():.4f}")
    print(f"\nFinal blend weights:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        if name in available:
            print(f"  {name:30s}  {w:.3f}")


if __name__ == "__main__":
    main()
