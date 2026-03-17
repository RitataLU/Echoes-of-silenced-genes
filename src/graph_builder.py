"""
Build the gene interaction graph used by the GNN.

Two types of edges:
  1. STRING PPI (high-confidence, score >= STRING_SCORE_GNN)
  2. Co-expression from non-targeting control cells (Pearson |r| >= threshold,
     top-K edges per gene to keep the graph sparse)

Run this script directly to download external data and build the graph:
  python src/graph_builder.py --download-string --download-go
  python src/graph_builder.py --build
"""
from __future__ import annotations
import os
import sys
import argparse
import gzip
import requests
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_adata, get_control_stats, load_ground_truth, load_pert_ids, load_means


# ── STRING download ────────────────────────────────────────────────────────

def download_string_interactions(gene_list: List[str],
                                  score_threshold: int = config.STRING_SCORE_KNN,
                                  save_path: str = config.STRING_TSV_PATH) -> pd.DataFrame:
    """
    Download gene-gene interactions from STRING API for a given gene list.
    Splits into chunks of 1800 genes to respect API limits.
    Results are cached to save_path.

    Returns a DataFrame with columns: gene_a, gene_b, score (0–1000 scale).
    """
    if os.path.exists(save_path):
        print(f"Loading cached STRING interactions from {save_path} …")
        df = pd.read_csv(save_path, sep="\t")
        df = df[df["score"] >= score_threshold / 1000.0].reset_index(drop=True)
        print(f"  {len(df)} interactions loaded (score >= {score_threshold})")
        return df

    config.make_dirs()
    print(f"Downloading STRING interactions for {len(gene_list)} genes (score >= {score_threshold}) …")

    CHUNK_SIZE = 1800
    all_rows = []

    for start in range(0, len(gene_list), CHUNK_SIZE):
        chunk = gene_list[start:start + CHUNK_SIZE]
        params = {
            "identifiers"   : "\r".join(chunk),
            "species"       : config.STRING_SPECIES,
            "required_score": score_threshold,
            "network_type"  : "functional",
            "caller_identity": "kaggle_gears_competition",
        }
        resp = requests.post(
            "https://string-db.org/api/tsv/network",
            data=params,
            timeout=120,
        )
        resp.raise_for_status()

        lines = resp.text.strip().split("\n")
        if len(lines) <= 1:
            print(f"  Chunk {start}–{start+len(chunk)}: no interactions returned")
            continue

        header = lines[0].split("\t")
        for line in lines[1:]:
            fields = line.split("\t")
            row = dict(zip(header, fields))
            all_rows.append({
                "gene_a": row.get("preferredName_A", ""),
                "gene_b": row.get("preferredName_B", ""),
                "score" : float(row.get("score", 0)),
            })
        print(f"  Chunk {start}–{start+len(chunk)}: {len(lines)-1} interactions")

    df = pd.DataFrame(all_rows).drop_duplicates()
    # Remove self-loops
    df = df[df["gene_a"] != df["gene_b"]].reset_index(drop=True)
    df.to_csv(save_path, sep="\t", index=False)
    print(f"STRING interactions saved to {save_path} ({len(df)} edges)")
    return df


def load_string_interactions(score_threshold: int = config.STRING_SCORE_KNN) -> pd.DataFrame:
    """Load cached STRING interactions, filtered by score threshold."""
    df = pd.read_csv(config.STRING_TSV_PATH, sep="\t")
    return df[df["score"] >= score_threshold / 1000.0].reset_index(drop=True)


# ── Co-expression edges ────────────────────────────────────────────────────

def build_coexp_edges(adata, node_names: List[str],
                       threshold: float = config.STRING_COEXP_THRESHOLD,
                       top_k: int = config.STRING_COEXP_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sparse co-expression edges from non-targeting control cells.

    For memory efficiency, we compute co-expression only between genes in
    node_names, using the control cell expression matrix.

    Returns
    -------
    edge_index : (2, n_edges) int64 – source and target node indices
    edge_attr  : (n_edges,) float32 – absolute Pearson correlation
    """
    cache_ei = os.path.join(config.CACHE_DIR, "coexp_edge_index.npy")
    cache_ea = os.path.join(config.CACHE_DIR, "coexp_edge_attr.npy")

    if os.path.exists(cache_ei) and os.path.exists(cache_ea):
        print("Loading cached co-expression edges …")
        return np.load(cache_ei), np.load(cache_ea)

    print("Computing co-expression edges from control cells …")
    ctrl = adata[adata.obs["sgrna_symbol"] == "non-targeting"]

    # Filter to genes in node_names only
    gene_mask = adata.var_names.isin(set(node_names))
    gene_idx_in_adata = np.where(gene_mask)[0]
    filtered_names = adata.var_names[gene_idx_in_adata].tolist()
    gene_to_node = {g: node_names.index(g) for g in filtered_names if g in node_names}

    X = ctrl.X[:, gene_idx_in_adata]
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Standardize each gene
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - mu) / sd   # (n_cells, n_genes_subset)

    n_genes = X_norm.shape[1]
    src_list, dst_list, val_list = [], [], []

    BATCH = 500  # process 500 genes at a time to avoid OOM
    for start in tqdm(range(0, n_genes, BATCH), desc="Co-exp edges"):
        end = min(start + BATCH, n_genes)
        corr_block = (X_norm[:, start:end].T @ X_norm) / X_norm.shape[0]
        # corr_block shape: (end-start, n_genes)

        for local_i, i in enumerate(range(start, end)):
            row = np.abs(corr_block[local_i])   # (n_genes,)
            row[i] = 0.0                         # no self-loop
            # keep top-k above threshold
            candidates = np.where(row >= threshold)[0]
            if len(candidates) == 0:
                continue
            if len(candidates) > top_k:
                candidates = candidates[np.argsort(row[candidates])[-top_k:]]

            node_i = gene_to_node[filtered_names[i]]
            for j in candidates:
                node_j = gene_to_node.get(filtered_names[j])
                if node_j is None:
                    continue
                src_list.append(node_i)
                dst_list.append(node_j)
                val_list.append(float(row[j]))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = np.array(val_list, dtype=np.float32)

    np.save(cache_ei, edge_index)
    np.save(cache_ea, edge_attr)
    print(f"Co-expression edges: {edge_index.shape[1]}")
    return edge_index, edge_attr


# ── Main graph builder ─────────────────────────────────────────────────────

def build_graph(score_threshold: int = config.STRING_SCORE_GNN,
                adata=None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Build the combined gene interaction graph.

    Node set = all 5127 output genes ∪ all 120 test perturbation genes ∪ all 80 train genes.

    Returns
    -------
    node_names : list of gene names (length = n_nodes)
    edge_index : (2, n_edges) int64
    edge_attr  : (n_edges,) float32 – normalised scores (0–1)
    """
    cache_nn  = config.NODE_NAMES_PATH
    cache_ei  = os.path.join(config.CACHE_DIR, f"string_edge_index_{score_threshold}.npy")
    cache_ea  = os.path.join(config.CACHE_DIR, f"string_edge_attr_{score_threshold}.npy")

    # ── Node set ──────────────────────────────────────────────────────────
    # Load output genes (5127) and perturbation genes (200)
    _, _, gene_order, pert_order = load_ground_truth()
    test_df = load_pert_ids()
    test_genes = test_df["pert"].tolist()

    all_genes_set = set(gene_order) | set(pert_order) | set(test_genes)

    if os.path.exists(cache_nn):
        node_names = open(cache_nn).read().splitlines()
        # Extend if new genes were added
        existing = set(node_names)
        for g in all_genes_set:
            if g not in existing:
                node_names.append(g)
    else:
        node_names = sorted(all_genes_set)
        config.make_dirs()

    with open(cache_nn, "w") as f:
        f.write("\n".join(node_names))

    node_to_idx = {g: i for i, g in enumerate(node_names)}
    print(f"Node set: {len(node_names)} genes")

    # ── STRING edges ──────────────────────────────────────────────────────
    if os.path.exists(cache_ei) and os.path.exists(cache_ea):
        print(f"Loading cached STRING edges (score >= {score_threshold}) …")
        string_ei = np.load(cache_ei)
        string_ea = np.load(cache_ea)
    else:
        string_df = load_string_interactions(score_threshold=score_threshold)
        src_list, dst_list, val_list = [], [], []
        for _, row in string_df.iterrows():
            a, b = row["gene_a"], row["gene_b"]
            if a in node_to_idx and b in node_to_idx:
                i, j = node_to_idx[a], node_to_idx[b]
                # Add both directions (undirected graph)
                src_list += [i, j]
                dst_list += [j, i]
                val_list += [float(row["score"]), float(row["score"])]

        string_ei = np.array([src_list, dst_list], dtype=np.int64)
        string_ea = np.array(val_list, dtype=np.float32)
        np.save(cache_ei, string_ei)
        np.save(cache_ea, string_ea)
    print(f"STRING edges: {string_ei.shape[1]}")

    # ── Co-expression edges ───────────────────────────────────────────────
    if adata is None:
        from data_utils import load_adata
        adata = load_adata()

    coexp_ei, coexp_ea = build_coexp_edges(adata, node_names)

    # ── Merge edges ───────────────────────────────────────────────────────
    # Normalise STRING scores to [0, 1] (they're already 0–1 from API)
    # Co-expression scores are already |r| in [0, 1]
    edge_index = np.concatenate([string_ei, coexp_ei], axis=1)
    edge_attr  = np.concatenate([string_ea, coexp_ea], axis=0)

    # Remove duplicate edges (keep max score per pair)
    pairs = {}
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        key = (i, j)
        pairs[key] = max(pairs.get(key, 0.0), float(edge_attr[k]))

    srcs = np.array([k[0] for k in pairs], dtype=np.int64)
    dsts = np.array([k[1] for k in pairs], dtype=np.int64)
    vals = np.array(list(pairs.values()), dtype=np.float32)

    edge_index = np.stack([srcs, dsts], axis=0)
    edge_attr  = vals

    print(f"Final graph: {len(node_names)} nodes, {edge_index.shape[1]} edges")
    return node_names, edge_index, edge_attr


# ── GO file download ───────────────────────────────────────────────────────

def download_go_files():
    """Download GO OBO file and human GOA annotation file."""
    config.make_dirs()

    # GO OBO
    obo_url = "https://purl.obolibrary.org/obo/go/go-basic.obo"
    if not os.path.exists(config.GO_OBO_PATH):
        print(f"Downloading {obo_url} …")
        resp = requests.get(obo_url, timeout=120)
        resp.raise_for_status()
        with open(config.GO_OBO_PATH, "wb") as f:
            f.write(resp.content)
        print(f"Saved to {config.GO_OBO_PATH}")
    else:
        print(f"GO OBO already exists: {config.GO_OBO_PATH}")

    # Human GOA
    gaf_url = ("https://current.geneontology.org/annotations/goa_human.gaf.gz")
    if not os.path.exists(config.GO_GAF_PATH):
        print(f"Downloading {gaf_url} …")
        resp = requests.get(gaf_url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(config.GO_GAF_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {config.GO_GAF_PATH}")
    else:
        print(f"Human GOA already exists: {config.GO_GAF_PATH}")


# ── String interactions for KNN (lower threshold) ─────────────────────────

def get_string_scores_for_knn(gene_list: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Return a dict {(gene_a, gene_b): score} for KNN use.
    Scores are in 0–1 range (divided by 1000 from STRING's 0–1000 scale).
    Both directions are stored.
    """
    df = download_string_interactions(gene_list, score_threshold=config.STRING_SCORE_KNN)
    scores = {}
    for _, row in df.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        s = float(row["score"])
        scores[(a, b)] = max(scores.get((a, b), 0.0), s)
        scores[(b, a)] = max(scores.get((b, a), 0.0), s)
    return scores


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build gene interaction graph")
    parser.add_argument("--download-string", action="store_true",
                        help="Download STRING PPI interactions via API")
    parser.add_argument("--download-go", action="store_true",
                        help="Download GO OBO and human GOA files")
    parser.add_argument("--build", action="store_true",
                        help="Build the full gene graph (STRING + co-expression)")
    parser.add_argument("--score", type=int, default=config.STRING_SCORE_GNN,
                        help=f"STRING score threshold for GNN graph (default {config.STRING_SCORE_GNN})")
    args = parser.parse_args()

    if args.download_string:
        _, _, gene_order, pert_order = load_ground_truth()
        test_df = load_pert_ids()
        all_genes = list(set(gene_order) | set(pert_order) | set(test_df["pert"].tolist()))
        download_string_interactions(all_genes, score_threshold=config.STRING_SCORE_KNN)

    if args.download_go:
        download_go_files()

    if args.build:
        node_names, edge_index, edge_attr = build_graph(score_threshold=args.score)
        print(f"\nGraph summary:")
        print(f"  Nodes: {len(node_names)}")
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Avg degree: {edge_index.shape[1] / len(node_names):.1f}")


if __name__ == "__main__":
    main()
