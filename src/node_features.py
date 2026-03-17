"""
Build the node feature matrix for all genes in the graph.

Three components, concatenated per gene:
  1. Expression statistics (5D)    – from non-targeting control cells
  2. Co-expression PCA (50D)       – TruncatedSVD on control expression matrix
  3. GO term embeddings (64D)      – TruncatedSVD on binary gene × GO-term matrix

Final matrix: (n_nodes, 5+50+64) = (n_nodes, 119) float32
The GNN input projection layer maps this to GNN_PROJ_DIM (128).

Run directly to build and cache the feature matrix:
  python src/node_features.py
"""
from __future__ import annotations
import os
import sys
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_adata, get_control_stats


# ── Component 1: Expression statistics ────────────────────────────────────

def get_expr_stats_for_nodes(node_names: List[str],
                              adata=None) -> np.ndarray:
    """
    Return expression stats for each node gene.
    Genes not in the h5ad panel get zeros.

    Returns: (n_nodes, 5) float32
    """
    if adata is None:
        adata = load_adata()

    stats_all, gene_names_all = get_control_stats(adata)
    gene_to_idx = {g: i for i, g in enumerate(gene_names_all)}

    out = np.zeros((len(node_names), config.EXPR_STATS_DIM), dtype=np.float32)
    for j, gene in enumerate(node_names):
        if gene in gene_to_idx:
            out[j] = stats_all[gene_to_idx[gene]]
    return out


# ── Component 2: Co-expression PCA ────────────────────────────────────────

def get_coexp_pca_for_nodes(node_names: List[str],
                             adata=None) -> np.ndarray:
    """
    Compute TruncatedSVD on transposed control expression matrix:
    (n_genes, n_ctrl_cells) → top-50 singular vectors per gene.

    The resulting coordinates encode co-expression structure.
    Genes not in the h5ad panel get zeros.

    Returns: (n_nodes, COEXP_PCA_DIM) float32
    """
    cache = config.COEXP_EMBED_PATH
    cache_names = os.path.join(config.CACHE_DIR, "coexp_gene_names.txt")

    if os.path.exists(cache):
        print("Loading cached co-expression PCA …")
        embed = np.load(cache)
        embed_gene_names = open(cache_names).read().splitlines()
    else:
        if adata is None:
            adata = load_adata()

        ctrl = adata[adata.obs["sgrna_symbol"] == "non-targeting"]
        X = ctrl.X  # (n_ctrl_cells, n_genes)
        if sp.issparse(X):
            X = X.toarray()
        X = X.astype(np.float32)

        # X.T shape: (n_genes, n_ctrl_cells)
        print(f"Running TruncatedSVD on control matrix ({adata.n_vars} genes × {X.shape[0]} cells) …")
        svd = TruncatedSVD(n_components=config.COEXP_PCA_DIM, random_state=config.SEED)
        embed = svd.fit_transform(X.T)  # (n_genes, COEXP_PCA_DIM)
        embed = embed.astype(np.float32)
        embed_gene_names = adata.var_names.tolist()

        config.make_dirs()
        np.save(cache, embed)
        with open(cache_names, "w") as f:
            f.write("\n".join(embed_gene_names))
        print(f"Co-expression PCA cached: {embed.shape}")

    gene_to_idx = {g: i for i, g in enumerate(embed_gene_names)}
    out = np.zeros((len(node_names), config.COEXP_PCA_DIM), dtype=np.float32)
    for j, gene in enumerate(node_names):
        if gene in gene_to_idx:
            out[j] = embed[gene_to_idx[gene]]
    return out


# ── Component 3: GO term embeddings ───────────────────────────────────────

def parse_gaf(gaf_path: str) -> Dict[str, set]:
    """
    Parse a GO Annotation File (GAF format, possibly gzipped).
    Returns dict {gene_symbol: set of GO term IDs}.
    """
    gene_go: Dict[str, set] = {}
    opener = gzip.open if gaf_path.endswith(".gz") else open

    with opener(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            symbol  = parts[2]   # gene symbol (column 3)
            go_id   = parts[4]   # GO ID (column 5)
            qualifier = parts[3] # NOT annotations — skip
            if "NOT" in qualifier:
                continue
            gene_go.setdefault(symbol, set()).add(go_id)

    return gene_go


def propagate_go_ancestors(go_terms: set, go_graph) -> set:
    """Add all ancestor GO terms for a given set of terms (term propagation)."""
    import networkx as nx
    ancestors = set(go_terms)
    for term in go_terms:
        if term in go_graph:
            ancestors.update(nx.ancestors(go_graph, term))
    return ancestors


def build_go_embeddings(node_names: List[str]) -> np.ndarray:
    """
    Build GO term embeddings for node genes via TruncatedSVD on a
    binary gene × GO-term matrix with ancestor propagation.

    Returns: (n_nodes, GO_EMBED_DIM) float32
    """
    cache = config.GO_EMBED_PATH
    cache_names = os.path.join(config.CACHE_DIR, "go_gene_names.txt")

    if os.path.exists(cache):
        print("Loading cached GO embeddings …")
        embed = np.load(cache)
        embed_gene_names = open(cache_names).read().splitlines()
    else:
        if not os.path.exists(config.GO_OBO_PATH):
            print("GO OBO not found — skipping GO embeddings (run graph_builder.py --download-go)")
            return np.zeros((len(node_names), config.GO_EMBED_DIM), dtype=np.float32)
        if not os.path.exists(config.GO_GAF_PATH):
            print("GO GAF not found — skipping GO embeddings (run graph_builder.py --download-go)")
            return np.zeros((len(node_names), config.GO_EMBED_DIM), dtype=np.float32)

        try:
            import obonet
        except ImportError:
            print("obonet not installed — skipping GO embeddings (pip install obonet)")
            return np.zeros((len(node_names), config.GO_EMBED_DIM), dtype=np.float32)

        print("Parsing GO ontology (OBO) …")
        go_graph = obonet.read_obo(config.GO_OBO_PATH)

        print("Parsing human GO annotations (GAF) …")
        gene_go = parse_gaf(config.GO_GAF_PATH)

        print("Propagating GO ancestors …")
        # Use only the node_names gene set for the matrix (for speed)
        node_set = set(node_names)
        gene_go_prop = {}
        for gene in tqdm(node_names, desc="GO propagation"):
            raw = gene_go.get(gene, set())
            gene_go_prop[gene] = propagate_go_ancestors(raw, go_graph)

        # Build GO term vocabulary (terms that appear in at least 2 genes)
        from collections import Counter
        all_terms = Counter()
        for terms in gene_go_prop.values():
            all_terms.update(terms)
        vocab = [t for t, cnt in all_terms.items() if cnt >= 2]
        term_to_idx = {t: i for i, t in enumerate(vocab)}
        print(f"GO vocabulary: {len(vocab)} terms")

        # Build sparse binary matrix: (n_genes, n_go_terms)
        rows, cols = [], []
        for gi, gene in enumerate(node_names):
            for term in gene_go_prop.get(gene, []):
                if term in term_to_idx:
                    rows.append(gi)
                    cols.append(term_to_idx[term])

        mat = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(node_names), len(vocab))
        )
        print(f"Gene × GO matrix: {mat.shape}, density: {mat.nnz / (mat.shape[0]*mat.shape[1]):.4f}")

        # TruncatedSVD to GO_EMBED_DIM
        n_components = min(config.GO_EMBED_DIM, min(mat.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=config.SEED)
        embed = svd.fit_transform(mat).astype(np.float32)  # (n_nodes, GO_EMBED_DIM)

        # Pad to GO_EMBED_DIM if needed
        if embed.shape[1] < config.GO_EMBED_DIM:
            pad = np.zeros((embed.shape[0], config.GO_EMBED_DIM - embed.shape[1]), dtype=np.float32)
            embed = np.concatenate([embed, pad], axis=1)

        embed_gene_names = node_names

        config.make_dirs()
        np.save(cache, embed)
        with open(cache_names, "w") as f:
            f.write("\n".join(embed_gene_names))
        print(f"GO embeddings cached: {embed.shape}")

    # Map to node_names order
    gene_to_idx = {g: i for i, g in enumerate(embed_gene_names)}
    out = np.zeros((len(node_names), config.GO_EMBED_DIM), dtype=np.float32)
    for j, gene in enumerate(node_names):
        if gene in gene_to_idx:
            out[j] = embed[gene_to_idx[gene]]
    return out


# ── Final feature assembly ─────────────────────────────────────────────────

def build_node_features(node_names: List[str], adata=None) -> np.ndarray:
    """
    Assemble the full node feature matrix by concatenating all components.

    Returns: (n_nodes, NODE_FEAT_DIM) float32  where NODE_FEAT_DIM = 119
    """
    cache = config.NODE_FEATURES_PATH

    if os.path.exists(cache):
        # Check if node set is the same
        cached_names_path = config.NODE_NAMES_PATH
        if os.path.exists(cached_names_path):
            cached_names = open(cached_names_path).read().splitlines()
            if cached_names == node_names:
                print("Loading cached node features …")
                return np.load(cache)

    config.make_dirs()

    print("Building node feature matrix …")
    if adata is None:
        adata = load_adata()

    feat_expr  = get_expr_stats_for_nodes(node_names, adata)   # (N, 5)
    feat_coexp = get_coexp_pca_for_nodes(node_names, adata)    # (N, 50)
    feat_go    = build_go_embeddings(node_names)               # (N, 64)

    features = np.concatenate([feat_expr, feat_coexp, feat_go], axis=1)

    # L2-normalise each feature component separately to prevent scale issues
    def _l2norm(x):
        norms = np.linalg.norm(x, axis=0, keepdims=True) + 1e-8
        return x / norms

    features = np.concatenate([
        _l2norm(feat_expr),
        _l2norm(feat_coexp),
        _l2norm(feat_go),
    ], axis=1).astype(np.float32)

    np.save(cache, features)
    print(f"Node features: {features.shape}")
    return features


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from graph_builder import build_graph

    node_names, _, _ = build_graph()
    features = build_node_features(node_names)

    print(f"\nNode feature matrix: {features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std:  {features.std():.4f}")
    print(f"  NaN:  {np.isnan(features).sum()}")
