"""
Phase 1: KNN Baseline using STRING PPI similarity.

For each test gene, find the top-k most similar training genes by STRING score,
then predict their DE as a softmax-weighted average.

Fallback for genes with no STRING connections: use co-expression similarity
computed from the non-targeting control cells in the h5ad file.

Usage:
    # First download STRING data (one-time):
    python src/graph_builder.py --download-string

    # Then generate the KNN submission:
    python src/knn_baseline.py
    # → outputs/submissions/submission_knn.csv
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import (
    load_ground_truth, load_pert_ids, load_sample_submission,
    load_adata, get_control_stats, compute_wmae
)
from graph_builder import download_string_interactions


# ── STRING similarity matrix ───────────────────────────────────────────────

def build_string_similarity(all_genes: list[str],
                             train_genes: list[str]) -> np.ndarray:
    """
    Build a (n_all, n_train) similarity matrix from STRING scores.

    all_genes  : ordered list of query genes (120 test perts)
    train_genes: ordered list of training genes (80)

    Returns: (120, 80) float32 — STRING score in [0, 1] range
    """
    string_df = download_string_interactions(
        gene_list=list(set(all_genes) | set(train_genes)),
        score_threshold=config.STRING_SCORE_KNN,
    )

    # Build lookup dict: (a, b) → score (0–1 scale)
    scores = {}
    for _, row in string_df.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        s = float(row["score"])
        scores[(a, b)] = max(scores.get((a, b), 0.0), s)
        scores[(b, a)] = max(scores.get((b, a), 0.0), s)

    train_idx = {g: i for i, g in enumerate(train_genes)}
    sim = np.zeros((len(all_genes), len(train_genes)), dtype=np.float32)

    for i, qg in enumerate(all_genes):
        for j, tg in enumerate(train_genes):
            sim[i, j] = scores.get((qg, tg), 0.0)

    print(f"STRING similarity matrix: {sim.shape}")
    print(f"  Genes with at least 1 STRING hit: {(sim.max(axis=1) > 0).sum()} / {len(all_genes)}")
    return sim


def build_coexp_similarity(query_genes: list[str],
                            train_genes: list[str],
                            adata=None) -> np.ndarray:
    """
    Compute co-expression similarity between query and train genes using the
    non-targeting control cells in h5ad.

    Returns: (n_query, n_train) float32 — absolute Pearson correlation
    """
    if adata is None:
        adata = load_adata()

    ctrl = adata[adata.obs["sgrna_symbol"] == "non-targeting"]
    import scipy.sparse as sp
    X = ctrl.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)  # (n_ctrl_cells, n_all_genes)

    gene_names = adata.var_names.tolist()
    gene_to_col = {g: i for i, g in enumerate(gene_names)}

    def get_profile(gene):
        if gene in gene_to_col:
            col = X[:, gene_to_col[gene]]
            return (col - col.mean()) / (col.std() + 1e-8)
        return None

    sim = np.zeros((len(query_genes), len(train_genes)), dtype=np.float32)
    n_cells = X.shape[0]

    for i, qg in enumerate(query_genes):
        qp = get_profile(qg)
        if qp is None:
            continue
        for j, tg in enumerate(train_genes):
            tp = get_profile(tg)
            if tp is None:
                continue
            sim[i, j] = abs(float(np.dot(qp, tp)) / n_cells)

    return sim


# ── KNN prediction ─────────────────────────────────────────────────────────

def knn_predict(sim_matrix: np.ndarray,
                de_matrix: np.ndarray,
                k: int = config.KNN_K,
                temperature: float = 0.5) -> np.ndarray:
    """
    KNN prediction using softmax-weighted average of training DE vectors.

    Parameters
    ----------
    sim_matrix : (n_query, n_train) similarity scores
    de_matrix  : (n_train, n_genes) ground truth DE
    k          : number of neighbours
    temperature: temperature for softmax (lower = more uniform, higher = more peaked).
                 Divides similarity scores before softmax to control weight concentration.

    Returns
    -------
    preds : (n_query, n_genes) predicted DE
    """
    n_query, n_train = sim_matrix.shape
    n_genes = de_matrix.shape[1]
    preds = np.zeros((n_query, n_genes), dtype=np.float32)

    for i in range(n_query):
        row = sim_matrix[i]
        # If all zeros: predict mean DE (the baseline)
        if row.max() == 0:
            preds[i] = de_matrix.mean(axis=0)
            continue

        # Top-k neighbours
        top_k_idx = np.argsort(row)[-k:]
        top_k_sim = row[top_k_idx]

        # For very weak matches, fall back to training mean
        if top_k_sim.max() < 0.05:
            preds[i] = de_matrix.mean(axis=0)
            continue

        # Softmax weights
        w = softmax(top_k_sim / temperature)
        preds[i] = (w[:, None] * de_matrix[top_k_idx]).sum(axis=0)

    # Clip predictions to per-gene min/max observed in training to prevent
    # outlier neighbours from producing unrealistic values
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max)

    return preds


# ── Main ───────────────────────────────────────────────────────────────────

def run_knn_baseline(k: int = config.KNN_K,
                     use_coexp_fallback: bool = True) -> str:
    """
    Build KNN predictions for all 120 test perturbations and save submission.

    Returns path to the submission CSV.
    """
    config.make_dirs()

    # Load data
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    test_df = load_pert_ids()
    test_genes_ordered = test_df["pert"].tolist()    # 120 genes in submission order
    pert_ids_ordered   = test_df["pert_id"].tolist() # corresponding pert_ids

    # Build STRING similarity
    sim_string = build_string_similarity(
        all_genes=test_genes_ordered,
        train_genes=train_genes,
    )

    # Co-expression fallback for genes with no STRING hits
    if use_coexp_fallback:
        no_hit_mask = sim_string.max(axis=1) == 0
        n_no_hit = no_hit_mask.sum()
        if n_no_hit > 0:
            print(f"\n{n_no_hit} genes with no STRING hits — computing co-expression similarity …")
            query_no_hit = [test_genes_ordered[i] for i in np.where(no_hit_mask)[0]]
            sim_coexp = build_coexp_similarity(query_no_hit, train_genes)
            # Insert co-expression similarity into the full matrix
            no_hit_indices = np.where(no_hit_mask)[0]
            for local_i, global_i in enumerate(no_hit_indices):
                sim_string[global_i] = sim_coexp[local_i]
            print(f"  After fallback, genes with ≥1 hit: {(sim_string.max(axis=1) > 0).sum()} / {len(test_genes_ordered)}")

    # KNN prediction
    print(f"\nRunning KNN (k={k}) …")
    preds = knn_predict(sim_string, de_matrix, k=k, temperature=0.5)
    print(f"Predictions shape: {preds.shape}")

    # Evaluate on training set (self-evaluation using leave-one-out would be ideal,
    # but here we just evaluate the KNN's ability to retrieve training genes)
    # For cross-validation: evaluate how well training genes predict each other
    print("\nKNN cross-evaluation on training set (k=5 neighbours from remaining 79):")
    train_sim = np.zeros((len(train_genes), len(train_genes)), dtype=np.float32)
    string_df = download_string_interactions(
        gene_list=train_genes,
        score_threshold=config.STRING_SCORE_KNN,
    )
    scores = {}
    for _, row in string_df.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        s = float(row["score"])
        scores[(a, b)] = max(scores.get((a, b), 0.0), s)
        scores[(b, a)] = max(scores.get((b, a), 0.0), s)
    for i, g1 in enumerate(train_genes):
        for j, g2 in enumerate(train_genes):
            if i != j:
                train_sim[i, j] = scores.get((g1, g2), 0.0)

    # Leave-one-out cross-validation on training set
    loo_preds = np.zeros_like(de_matrix)
    for i in range(len(train_genes)):
        sim_row = train_sim[i].copy()
        sim_row[i] = 0.0  # exclude self
        top_k = np.argsort(sim_row)[-k:]
        top_sim = sim_row[top_k]
        if top_sim.max() == 0 or top_sim.max() < 0.05:
            loo_preds[i] = de_matrix.mean(axis=0)
        else:
            w = softmax(top_sim / 0.5)
            loo_preds[i] = (w[:, None] * de_matrix[top_k]).sum(axis=0)

    loo_wmae = compute_wmae(loo_preds, de_matrix, weight_matrix)
    print(f"  LOO WMAE: {loo_wmae:.4f}  (baseline mean-DE: 0.1268)")

    # Build submission
    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", pert_ids_ordered)

    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_knn.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nKNN submission saved to: {out_path}")

    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.KNN_K)
    parser.add_argument("--no-coexp-fallback", action="store_true")
    args = parser.parse_args()

    path = run_knn_baseline(
        k=args.k,
        use_coexp_fallback=not args.no_coexp_fallback,
    )
    print(f"\nDone! Submit: {path}")
