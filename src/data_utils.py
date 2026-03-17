"""
Data loading and utility functions.
All heavy computation results are cached to disk to avoid re-running.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Tuple, Dict, List

import config


# ── Loaders ────────────────────────────────────────────────────────────────

def load_adata() -> ad.AnnData:
    """Load the training single-cell AnnData object."""
    print("Loading h5ad …")
    adata = ad.read_h5ad(config.H5AD_PATH)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_ground_truth() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the ground truth table.

    Returns
    -------
    de_matrix    : (80, 5127) float32 – differential expression values
    weight_matrix: (80, 5127) float32 – per-perturbation per-gene weights
    gene_order   : list[str] length 5127 – gene column names
    pert_order   : list[str] length 80   – perturbed gene names (row order)
    """
    df = pd.read_csv(config.GT_PATH)
    weight_cols = [c for c in df.columns if c.startswith("w_")]
    gene_cols   = [c for c in df.columns if not c.startswith("w_")
                   and c not in ("pert_id", "baseline_wmae")]

    de_matrix     = df[gene_cols].values.astype(np.float32)
    weight_matrix = df[weight_cols].values.astype(np.float32)
    pert_order    = df["pert_id"].tolist()
    gene_order    = gene_cols

    print(f"Ground truth: {de_matrix.shape[0]} perturbations × {de_matrix.shape[1]} genes")
    return de_matrix, weight_matrix, gene_order, pert_order


def load_means() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load training_data_means.csv.

    Returns
    -------
    means_df  : DataFrame with 'pert_symbol' index and gene columns
    gene_order: list of gene column names (5127)
    """
    df = pd.read_csv(config.MEANS_PATH)
    gene_order = [c for c in df.columns if c != "pert_symbol"]
    df = df.set_index("pert_symbol")
    return df, gene_order


def load_pert_ids() -> pd.DataFrame:
    """Load pert_ids_all.csv: columns pert, class, pert_id."""
    return pd.read_csv(config.PERT_IDS_ALL_PATH)


def load_sample_submission() -> pd.DataFrame:
    """Load sample submission template."""
    return pd.read_csv(config.SAMPLE_SUB_PATH)


# ── Control cell statistics ────────────────────────────────────────────────

def get_control_stats(adata: ad.AnnData) -> Tuple[np.ndarray, List[str]]:
    """
    Compute per-gene statistics from non-targeting control cells.

    Returns
    -------
    stats   : (n_genes, 5) float32 array
              columns: [mean, log1p_mean, variance, dispersion, detection_rate]
    gene_names: list of gene names matching the column order of stats
    """
    cache_path = os.path.join(config.CACHE_DIR, "ctrl_stats.npy")
    names_path = os.path.join(config.CACHE_DIR, "ctrl_gene_names.txt")

    if os.path.exists(cache_path):
        print("Loading cached control stats …")
        stats = np.load(cache_path)
        gene_names = open(names_path).read().splitlines()
        return stats, gene_names

    config.make_dirs()
    ctrl = adata[adata.obs["sgrna_symbol"] == "non-targeting"]
    X = ctrl.X  # sparse (n_ctrl_cells, n_genes)
    if sp.issparse(X):
        X = X.toarray()

    mean      = X.mean(axis=0)                                   # (n_genes,)
    log_mean  = np.log1p(mean)
    var       = X.var(axis=0)
    disp      = np.where(mean > 0, var / mean, 0.0)
    detect    = (X > 0).mean(axis=0)

    stats = np.stack([mean, log_mean, var, disp, detect], axis=1).astype(np.float32)
    gene_names = adata.var_names.tolist()

    np.save(cache_path, stats)
    with open(names_path, "w") as f:
        f.write("\n".join(gene_names))

    print(f"Control stats computed: {stats.shape}")
    return stats, gene_names


# ── Evaluation metric ──────────────────────────────────────────────────────

def compute_wmae(pred: np.ndarray, gt: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute Weighted Mean Absolute Error matching the competition metric.

    Parameters
    ----------
    pred    : (n_perts, n_genes) predictions
    gt      : (n_perts, n_genes) ground truth
    weights : (n_perts, n_genes) per-perturbation per-gene weights

    Returns
    -------
    scalar WMAE
    """
    return float(np.mean(weights * np.abs(pred - gt)))


def compute_wmae_per_pert(pred: np.ndarray, gt: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
    """Return WMAE for each perturbation separately. Shape: (n_perts,)"""
    return np.mean(weights * np.abs(pred - gt), axis=1)


# ── Gene list helpers ──────────────────────────────────────────────────────

def get_all_pert_genes() -> Dict[str, str]:
    """
    Return dict {gene_name: class} for all 120 test perturbations.
    class is 'val' or 'test'.
    """
    df = load_pert_ids()
    return dict(zip(df["pert"], df["class"]))


def get_train_genes() -> List[str]:
    """Return the 80 training gene names (matching GT row order)."""
    _, _, _, pert_order = load_ground_truth()
    return pert_order


def get_test_genes() -> pd.DataFrame:
    """Return pert_ids_all DataFrame for all 120 test genes."""
    return load_pert_ids()


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    config.make_dirs()

    adata = load_adata()
    de, weights, gene_order, pert_order = load_ground_truth()
    means, _ = load_means()
    perts = load_pert_ids()

    print(f"\nDE matrix:     {de.shape}")
    print(f"Weight matrix: {weights.shape}")
    print(f"Gene order:    {gene_order[:3]} … {gene_order[-3:]}")
    print(f"Pert order:    {pert_order[:3]} … {pert_order[-3:]}")
    print(f"\nTest perts:    {len(perts)} ({perts['class'].value_counts().to_dict()})")

    # Verify GT formula: DE = means[pert] - means['non-targeting']
    nt_mean = means.loc["non-targeting", gene_order].values
    for gene in pert_order[:3]:
        pred = means.loc[gene, gene_order].values - nt_mean
        gt   = de[pert_order.index(gene)]
        assert np.allclose(pred, gt, atol=1e-5), f"GT mismatch for {gene}"
    print("\nGT formula verified: DE = means[pert] - means[non-targeting] ✓")

    # Baseline WMAE
    baseline_pred = np.tile(de.mean(axis=0, keepdims=True), (de.shape[0], 1))
    wmae = compute_wmae(baseline_pred, de, weights)
    print(f"Baseline WMAE (mean DE): {wmae:.4f}  (expect ~0.1268)")

    # Zero predictor WMAE
    zero_pred = np.zeros_like(de)
    wmae_zero = compute_wmae(zero_pred, de, weights)
    print(f"Zero predictor WMAE:     {wmae_zero:.4f}  (expect ~0.1413)")

    stats, gene_names = get_control_stats(adata)
    print(f"\nControl stats: {stats.shape} ({len(gene_names)} genes)")
