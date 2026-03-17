"""
Gene Program decomposition via Truncated SVD.

Key idea
--------
The DE matrix (80 perturbations × 5127 genes) is low-rank in practice:
    DE ≈ W @ H
where
    W : (n_perts, K)   — per-perturbation "program scores"
    H : (K, 5127)      — K gene programs (which genes co-vary together)

Instead of predicting 5127 raw DE values, the GNN predicts K program scores.
At inference:
    de_pred = predicted_W @ H     shape: (5127,)

Benefits
--------
1. Dramatically reduces output dimensionality (5127 → 64).
2. Constrains predictions to the subspace spanned by training data.
3. Biologically meaningful: each program captures a co-regulated gene module
   (e.g., cell-cycle genes, immune genes, mitochondrial genes).
4. Better generalisation for zero-shot genes: the model only needs to place
   the new perturbation in a 64D space, not predict 5127 values independently.

Weighted SVD
-----------
We weight each gene by its mean evaluation weight (from the GT table) before
decomposition, so high-weight genes drive the program structure.

Usage
-----
    from gene_programs import build_gene_programs, decode_programs
    H, gene_order, svd = build_gene_programs(de_matrix, weight_matrix, gene_order)
    de_pred = decode_programs(w_pred, H)   # (n_genes,)
"""
from __future__ import annotations
import os
import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth


# ── Build programs ─────────────────────────────────────────────────────────

def build_gene_programs(
    de_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    gene_order: list[str],
    K: int = config.GENE_PROGRAM_K,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    """
    Decompose the DE matrix into K gene programs using weighted SVD.

    Parameters
    ----------
    de_matrix     : (n_train, n_genes) float32 — ground-truth DE
    weight_matrix : (n_train, n_genes) float32 — evaluation weights
    gene_order    : list of n_genes gene names (column order)
    K             : number of gene programs

    Returns
    -------
    W   : (n_train, K)  — per-perturbation program scores (training)
    H   : (K, n_genes)  — gene program matrix
    svd : fitted TruncatedSVD object (for projecting new perturbations)
    """
    # Per-gene importance weight = mean weight across all perturbations
    gene_weights = weight_matrix.mean(axis=0)          # (n_genes,)
    gene_weights = gene_weights / (gene_weights.sum() + 1e-8)

    # Weight the DE matrix columns by sqrt(gene_weight) before SVD,
    # so high-weight genes contribute more to the program structure.
    sqrt_w = np.sqrt(gene_weights)[None, :]            # (1, n_genes)
    DE_weighted = de_matrix * sqrt_w                   # (n_train, n_genes)

    K_actual = min(K, min(DE_weighted.shape) - 1)
    svd = TruncatedSVD(n_components=K_actual, random_state=seed)
    W = svd.fit_transform(DE_weighted)                 # (n_train, K)

    # H in the weighted space; un-weight columns to get predictions in DE space
    H_weighted = svd.components_                       # (K, n_genes)
    inv_sqrt_w = 1.0 / (sqrt_w + 1e-8)
    H = H_weighted * inv_sqrt_w                        # (K, n_genes)

    var_explained = svd.explained_variance_ratio_.sum()
    print(f"Gene programs: K={K_actual}, variance explained: {var_explained:.1%}")

    return W.astype(np.float32), H.astype(np.float32), svd


def decode_programs(w: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Reconstruct DE from program scores.

    Parameters
    ----------
    w : (K,) or (batch, K)
    H : (K, n_genes)

    Returns
    -------
    de : (n_genes,) or (batch, n_genes)
    """
    return w @ H


# ── Cache helpers ──────────────────────────────────────────────────────────

def get_or_build_programs(
    de_matrix: np.ndarray | None = None,
    weight_matrix: np.ndarray | None = None,
    gene_order: list[str] | None = None,
    K: int = config.GENE_PROGRAM_K,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load gene programs from cache or build and cache them.

    Returns
    -------
    W          : (n_train, K) training program scores
    H          : (K, n_genes) gene program matrix
    gene_order : list of gene names (column order of H)
    """
    H_path     = config.GENE_PROGRAM_H
    names_path = config.GENE_PROGRAM_NAMES
    W_path     = os.path.join(config.CACHE_DIR, "gene_program_W.npy")

    if not force_rebuild and os.path.exists(H_path) and os.path.exists(W_path):
        print("Loading cached gene programs …")
        H          = np.load(H_path)
        W          = np.load(W_path)
        gene_order = open(names_path).read().splitlines()
        print(f"  W: {W.shape}, H: {H.shape}")
        return W, H, gene_order

    # Build from scratch
    if de_matrix is None:
        de_matrix, weight_matrix, gene_order, _ = load_ground_truth()

    config.make_dirs()
    W, H, _ = build_gene_programs(de_matrix, weight_matrix, gene_order, K=K)

    np.save(H_path, H)
    np.save(W_path, W)
    with open(names_path, "w") as f:
        f.write("\n".join(gene_order))

    print(f"Gene programs cached: W={W.shape}, H={H.shape}")
    return W, H, gene_order


# ── Weighted MAE in program space ──────────────────────────────────────────

def program_wmae_loss(
    w_pred: "torch.Tensor",
    w_true: "torch.Tensor",
    H: "torch.Tensor",
    weights: "torch.Tensor",
) -> "torch.Tensor":
    """
    Decode program scores to DE space and compute weighted MAE.

    Parameters
    ----------
    w_pred  : (K,)      predicted program scores
    w_true  : (K,)      ground-truth program scores (from SVD)
    H       : (K, G)    gene program matrix (on same device)
    weights : (G,)      per-gene evaluation weights

    Loss is computed in DE space (not program space) so it matches
    the competition metric exactly.
    """
    import torch
    de_pred = w_pred @ H      # (G,)
    de_true = w_true @ H      # (G,)
    return (weights * torch.abs(de_pred - de_true)).mean()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.GENE_PROGRAM_K)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    W, H, _ = get_or_build_programs(
        de_matrix, weight_matrix, gene_order,
        K=args.k, force_rebuild=args.rebuild,
    )

    # Reconstruction quality
    de_reconstructed = W @ H
    abs_err = np.abs(de_reconstructed - de_matrix)
    wmae = (abs_err * weight_matrix).mean()
    print(f"\nReconstruction WMAE (training, K={args.k}): {wmae:.4f}")
    print(f"  (Baseline mean-DE WMAE: 0.1268)")
    print(f"\nProgram score stats:")
    print(f"  W range: [{W.min():.3f}, {W.max():.3f}]")
    print(f"  H range: [{H.min():.3f}, {H.max():.3f}]")
