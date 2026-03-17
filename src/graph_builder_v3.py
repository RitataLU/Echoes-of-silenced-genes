"""
graph_builder_v3.py — Enhanced gene graph with directed TF regulatory edges.

Extends the v2 graph (STRING PPI + co-expression) by adding:
  - TF → target gene directed edges (one-directional, signal TF→target)

TF edge source priority:
  1. decoupler + DoRothEA (pip install decoupler) — ~170K pairs, A+B confidence
  2. TRRUST v2  (auto-downloaded TSV)            — ~8K pairs, human
  3. No TF edges (fallback with warning)

Edge attribute format changes from v2:
  v2: edge_attr (E,)       — scalar weight
  v3: edge_attr (E, 2)     — [weight (0–1), edge_type (float)]
        edge_type  0 = STRING PPI or co-expression (undirected)
        edge_type  1 = TF activating  (directed: TF → target)
        edge_type  2 = TF repressing  (directed: TF → target)

Usage
-----
    from graph_builder_v3 import build_graph_v3
    node_names, edge_index, edge_attr = build_graph_v3()
    # edge_attr.shape == (n_edges, 2)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import config
from graph_builder import build_graph   # reuse v2 base graph

# ── Cache paths ───────────────────────────────────────────────────────────

_TF_EI_CACHE  = os.path.join(config.CACHE_DIR, "tf_edge_index.npy")
_TF_EA_CACHE  = os.path.join(config.CACHE_DIR, "tf_edge_attr2d.npy")
_TRRUST_PATH  = os.path.join(config.EXTERNAL_DIR, "trrust_rawdata.human.tsv")
_TRRUST_URL   = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"


# ── TF edge loading ───────────────────────────────────────────────────────

def _load_dorothea(node_to_idx: dict) -> tuple[list, list, list]:
    """Try loading DoRothEA via decoupler. Returns (src, dst, type) lists."""
    import decoupler as dc
    print("Loading DoRothEA TF regulon (confidence A+B) via decoupler …")
    net = dc.get_dorothea(organism="human", levels=["A", "B"])
    # columns: source (TF), target, weight (1/-1), confidence
    src_list, dst_list, type_list = [], [], []
    n_skip = 0
    for _, row in net.iterrows():
        tf     = str(row["source"])
        target = str(row["target"])
        if tf not in node_to_idx or target not in node_to_idx:
            n_skip += 1
            continue
        etype = 1 if float(row["weight"]) > 0 else 2   # 1=activating 2=repressing
        src_list.append(node_to_idx[tf])
        dst_list.append(node_to_idx[target])
        type_list.append(etype)
    print(f"  DoRothEA: {len(src_list)} TF edges kept  ({n_skip} skipped — gene not in graph)")
    return src_list, dst_list, type_list


def _download_trrust() -> pd.DataFrame:
    """Download TRRUST v2 human TSV if not already cached."""
    if os.path.exists(_TRRUST_PATH):
        print(f"Loading cached TRRUST from {_TRRUST_PATH} …")
    else:
        config.make_dirs()
        print(f"Downloading TRRUST v2 from {_TRRUST_URL} …")
        resp = requests.get(_TRRUST_URL, timeout=60)
        resp.raise_for_status()
        with open(_TRRUST_PATH, "wb") as f:
            f.write(resp.content)
        print(f"  Saved → {_TRRUST_PATH}")

    df = pd.read_csv(
        _TRRUST_PATH, sep="\t", header=None,
        names=["tf", "target", "effect", "pubmed"],
    )
    return df


def _load_trrust(node_to_idx: dict) -> tuple[list, list, list]:
    """Load TRRUST v2 TF edges. Returns (src, dst, type) lists."""
    df = _download_trrust()
    src_list, dst_list, type_list = [], [], []
    n_skip = 0
    for _, row in df.iterrows():
        tf     = str(row["tf"])
        target = str(row["target"])
        if tf not in node_to_idx or target not in node_to_idx:
            n_skip += 1
            continue
        effect = str(row["effect"]).lower()
        if "repression" in effect:
            etype = 2
        elif "activation" in effect:
            etype = 1
        else:
            etype = 1   # "Unknown" → treat as activating
        src_list.append(node_to_idx[tf])
        dst_list.append(node_to_idx[target])
        type_list.append(etype)
    print(f"  TRRUST: {len(src_list)} TF edges kept  ({n_skip} skipped — gene not in graph)")
    return src_list, dst_list, type_list


def _build_tf_edges(node_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build TF regulatory edge arrays.

    Returns
    -------
    tf_ei : (2, E_tf) int64
    tf_ea : (E_tf, 2) float32  — [score=0.8, edge_type (1 or 2)]
    """
    if os.path.exists(_TF_EI_CACHE) and os.path.exists(_TF_EA_CACHE):
        print("Loading cached TF regulatory edges …")
        return np.load(_TF_EI_CACHE), np.load(_TF_EA_CACHE)

    node_to_idx = {g: i for i, g in enumerate(node_names)}

    # Try DoRothEA first, fall back to TRRUST
    src_list, dst_list, type_list = [], [], []
    try:
        src_list, dst_list, type_list = _load_dorothea(node_to_idx)
    except Exception as e:
        print(f"  decoupler/DoRothEA unavailable ({e}). Trying TRRUST v2 …")
        try:
            src_list, dst_list, type_list = _load_trrust(node_to_idx)
        except Exception as e2:
            print(f"  TRRUST also failed ({e2}). Continuing WITHOUT TF edges.")

    if not src_list:
        tf_ei = np.zeros((2, 0), dtype=np.int64)
        tf_ea = np.zeros((0, 2), dtype=np.float32)
    else:
        tf_ei = np.array([src_list, dst_list], dtype=np.int64)
        # Fixed confidence score for TF edges (TRRUST has no continuous score)
        tf_score = np.full(len(src_list), 0.8, dtype=np.float32)
        tf_type  = np.array(type_list, dtype=np.float32)
        tf_ea    = np.stack([tf_score, tf_type], axis=1)   # (E_tf, 2)

    config.make_dirs()
    np.save(_TF_EI_CACHE, tf_ei)
    np.save(_TF_EA_CACHE, tf_ea)
    print(f"TF edges cached: {tf_ei.shape[1]} edges")
    return tf_ei, tf_ea


# ── Main graph builder ─────────────────────────────────────────────────────

def build_graph_v3(
    score_threshold: int = config.STRING_SCORE_GNN,
    adata=None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Build enhanced gene graph (STRING + co-expression + TF regulatory).

    Returns
    -------
    node_names : list of str
    edge_index : (2, E) int64
    edge_attr  : (E, 2) float32 — [weight, edge_type]
                   edge_type: 0=PPI/coexp  1=TF_activating  2=TF_repressing
    """
    # ── Base graph (STRING + co-expression) from v2 ──────────────────────
    print("Building base graph (STRING + co-expression) …")
    node_names, base_ei, base_ea = build_graph(score_threshold, adata)
    # base_ea: (E,) — scores in [0, 1]
    # Promote to (E, 2) with edge_type = 0
    base_ea_2d = np.stack(
        [base_ea, np.zeros(len(base_ea), dtype=np.float32)],
        axis=1,
    )   # (E, 2)

    # ── TF regulatory edges ───────────────────────────────────────────────
    tf_ei, tf_ea = _build_tf_edges(node_names)

    # ── Merge ─────────────────────────────────────────────────────────────
    if tf_ei.shape[1] > 0:
        edge_index = np.concatenate([base_ei, tf_ei], axis=1)
        edge_attr  = np.concatenate([base_ea_2d, tf_ea], axis=0)
    else:
        edge_index = base_ei
        edge_attr  = base_ea_2d

    print(f"\nv3 Graph summary:")
    print(f"  Nodes : {len(node_names)}")
    print(f"  Edges : {edge_index.shape[1]}  "
          f"(base: {base_ei.shape[1]}, TF: {tf_ei.shape[1]})")
    print(f"  TF activating: {int((edge_attr[:, 1] == 1).sum())}  "
          f"repressing: {int((edge_attr[:, 1] == 2).sum())}")

    return node_names, edge_index, edge_attr


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node_names, ei, ea = build_graph_v3()
    print(f"\nedge_attr shape: {ea.shape}")
    print(f"edge_type distribution: {dict(zip(*np.unique(ea[:, 1].astype(int), return_counts=True)))}")
