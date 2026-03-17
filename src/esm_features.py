"""
ESM2 protein sequence embeddings for gene nodes.

Why ESM2?
---------
The current node features (119D) are computed from expression statistics,
co-expression PCA, and GO annotations — all derived from the training data.
ESM2 encodes evolutionary and structural information directly from the
protein sequence, which is a completely independent source of signal.

Two genes with similar protein sequences (e.g., paralogs, conserved domains)
will have similar ESM2 embeddings even if they don't co-express in this
dataset or share GO terms. This makes the node features richer and more
robust to missing database annotations.

Model choice
------------
- esm2_t6_8M_UR50D   : 8M params, 320D  — fastest, good baseline
- esm2_t12_35M_UR50D : 35M params, 480D — recommended balance
- esm2_t30_150M_UR50D: 150M params, 640D — best quality, needs ~4GB VRAM
- esm2_t33_650M_UR50D: 650M params, 1280D — best, needs ~8GB VRAM

Installation
------------
    pip install fair-esm
    # OR directly:
    pip install git+https://github.com/facebookresearch/esm.git

How it works
------------
1. For each gene name, fetch the canonical human protein sequence from UniProt.
2. Tokenise and run ESM2 forward pass → mean-pool the token embeddings.
3. Cache the result to avoid re-running.
4. Concatenate with existing node features to extend the 119D → (119 + esm_dim)D.

Usage
-----
    # One-time (slow — downloads sequences + runs ESM2 on GPU):
    python src/esm_features.py --model esm2_t12_35M_UR50D

    # Then retrain with ESM2 features (node_feat_dim updated automatically):
    python src/train_v2.py
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import requests
from typing import List

sys.path.insert(0, os.path.dirname(__file__))
import config

ESM_MODEL_NAME = "esm2_t12_35M_UR50D"   # 480D, good balance
ESM_EMBED_DIM  = 480                      # matches t12 model


# ── UniProt sequence fetch ─────────────────────────────────────────────────

def fetch_uniprot_sequence(gene_symbol: str,
                            organism_id: int = 9606,
                            retries: int = 3) -> str | None:
    """
    Fetch the canonical protein sequence for a human gene from UniProt REST API.
    Returns amino-acid string or None if not found.
    """
    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{gene_symbol}+AND+organism_id:{organism_id}"
        f"+AND+reviewed:true"
        f"&fields=sequence&format=fasta&size=1"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.text.strip():
                lines = resp.text.strip().split("\n")
                seq = "".join(l for l in lines if not l.startswith(">"))
                if seq:
                    return seq
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_sequences_bulk(gene_names: List[str],
                          cache_path: str) -> dict[str, str]:
    """
    Fetch sequences for all genes, caching results to avoid re-requests.

    Returns dict {gene_name: sequence_string}
    """
    import json

    # Load cache
    seqs = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            seqs = json.load(f)

    missing = [g for g in gene_names if g not in seqs]
    if missing:
        print(f"Fetching sequences for {len(missing)} genes from UniProt …")
        for i, gene in enumerate(missing):
            seq = fetch_uniprot_sequence(gene)
            seqs[gene] = seq or ""
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(missing)} …")
                # Save intermediate cache
                with open(cache_path, "w") as f:
                    json.dump(seqs, f)
            time.sleep(0.1)   # be polite to UniProt API

        with open(cache_path, "w") as f:
            json.dump(seqs, f)
        print(f"Sequences cached: {sum(1 for v in seqs.values() if v)} / {len(gene_names)} found")

    return seqs


# ── ESM2 embedding ─────────────────────────────────────────────────────────

def compute_esm2_embeddings(gene_names: List[str],
                             sequences: dict[str, str],
                             model_name: str = ESM_MODEL_NAME,
                             batch_size: int = 8,
                             device_str: str = "cuda") -> np.ndarray:
    """
    Compute mean-pooled ESM2 embeddings for a list of genes.

    Parameters
    ----------
    gene_names : list of gene names (defines output row order)
    sequences  : dict {gene_name: aa_sequence}
    model_name : ESM2 model variant
    batch_size : number of sequences per GPU batch
    device_str : "cuda" or "cpu"

    Returns
    -------
    embeddings : (n_genes, esm_dim) float32
                 Genes with no sequence get a zero vector.
    """
    try:
        import esm
    except ImportError:
        raise ImportError(
            "fair-esm not installed. Run: pip install fair-esm\n"
            "Or: pip install git+https://github.com/facebookresearch/esm.git"
        )

    import torch
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM2 model: {model_name} on {device} …")

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    esm_dim = model.embed_dim
    embeddings = np.zeros((len(gene_names), esm_dim), dtype=np.float32)

    # Only process genes with a sequence
    valid = [(i, g) for i, g in enumerate(gene_names) if sequences.get(g)]
    print(f"Computing ESM2 embeddings: {len(valid)} / {len(gene_names)} genes have sequences")

    import torch
    with torch.no_grad():
        for batch_start in range(0, len(valid), batch_size):
            batch = valid[batch_start:batch_start + batch_size]
            data  = [(g, sequences[g]) for _, g in batch]

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[model.num_layers],
                            return_contacts=False)
            token_reps = results["representations"][model.num_layers]

            for local_i, (global_i, gene) in enumerate(batch):
                # Mean pool over sequence length (exclude BOS/EOS tokens)
                seq_len  = len(sequences[gene])
                emb      = token_reps[local_i, 1:seq_len + 1].mean(0)
                embeddings[global_i] = emb.cpu().float().numpy()

            if (batch_start // batch_size + 1) % 10 == 0:
                print(f"  Batch {batch_start // batch_size + 1} / "
                      f"{(len(valid) + batch_size - 1) // batch_size}")

    return embeddings


# ── Build and cache ────────────────────────────────────────────────────────

def build_esm2_features(node_names: List[str],
                         model_name: str = ESM_MODEL_NAME,
                         force_rebuild: bool = False) -> np.ndarray:
    """
    Build or load ESM2 embeddings for all graph nodes.

    Returns
    -------
    embeddings : (n_nodes, esm_dim) float32
    """
    embed_path  = config.ESM2_EMBED_PATH
    names_path  = config.ESM2_GENE_NAMES
    seq_cache   = os.path.join(config.CACHE_DIR, "uniprot_sequences.json")
    config.make_dirs()

    if not force_rebuild and os.path.exists(embed_path):
        cached_names = open(names_path).read().splitlines()
        if cached_names == node_names:
            print("Loading cached ESM2 embeddings …")
            return np.load(embed_path)

    # Fetch sequences
    sequences = fetch_sequences_bulk(node_names, seq_cache)

    # Compute embeddings
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = compute_esm2_embeddings(
        node_names, sequences, model_name=model_name, device_str=device_str
    )

    # L2-normalise (same as other node feature components)
    norms = np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    np.save(embed_path, embeddings)
    with open(names_path, "w") as f:
        f.write("\n".join(node_names))
    print(f"ESM2 embeddings cached: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ── Integration helper ─────────────────────────────────────────────────────

def extend_node_features(base_features: np.ndarray,
                          node_names: List[str]) -> np.ndarray:
    """
    Append ESM2 embeddings to existing node features.

    Parameters
    ----------
    base_features : (n_nodes, 119)  — current hand-crafted features
    node_names    : list of gene names

    Returns
    -------
    extended : (n_nodes, 119 + esm_dim)  e.g. 119 + 480 = 599D
    """
    esm_features = build_esm2_features(node_names)
    extended = np.concatenate([base_features, esm_features], axis=1)
    print(f"Node features extended: {base_features.shape[1]}D → {extended.shape[1]}D")
    return extended.astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from graph_builder import build_graph
    from node_features import build_node_features

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default=ESM_MODEL_NAME)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    node_names, _, _ = build_graph()

    print(f"\nBuilding ESM2 features with {args.model} …")
    embeddings = build_esm2_features(node_names, model_name=args.model,
                                      force_rebuild=args.rebuild)
    print(f"ESM2 embedding shape: {embeddings.shape}")
    print(f"  Non-zero rows: {(embeddings.abs() if hasattr(embeddings, 'abs') else np.abs(embeddings)).sum(1).astype(bool).sum()} / {len(node_names)}")

    base = build_node_features(node_names)
    extended = extend_node_features(base, node_names)
    print(f"\nFinal node feature dim: {extended.shape[1]}")
    print("→ Update NODE_FEAT_DIM in config.py to match before training!")
