"""
GNN v2 inference — LOCAL (CPU/Mac) variant.

Loads v2_local_fold{k}_best.pt checkpoints produced by train_v2_local.py.
Output: submission_gnn_v2_local.csv  (does NOT overwrite GPU submission)

Usage
-----
    python src/predict_v2_local.py
    python src/predict_v2_local.py --tta 10
    python src/predict_v2_local.py --no-tta
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import config_local as config
from data_utils import load_ground_truth, load_pert_ids
from graph_builder import build_graph
from node_features import build_node_features
from data_utils import load_adata
from gene_programs import get_or_build_programs
from gnn_model_v2 import GEARSModelV2, mc_dropout_predict


def load_graph_to_device(device):
    adata = load_adata()
    node_names, edge_index, edge_attr = build_graph(
        score_threshold=config.STRING_SCORE_GNN, adata=adata
    )
    node_features = build_node_features(node_names, adata=adata)
    gene_to_node  = {g: i for i, g in enumerate(node_names)}

    _, _, gene_order, _ = load_ground_truth()
    output_gene_indices = [gene_to_node[g] for g in gene_order if g in gene_to_node]

    graph_x  = torch.FloatTensor(node_features).to(device)
    graph_ei = torch.LongTensor(edge_index).to(device)
    graph_ea = torch.FloatTensor(edge_attr).to(device)
    return graph_x, graph_ei, graph_ea, gene_to_node, gene_order, output_gene_indices


def predict_single_model(model, graph_x, graph_ei, graph_ea,
                         pert_gene_idx, output_gene_indices,
                         H_tensor, tta_samples, device):
    if tta_samples > 1:
        return mc_dropout_predict(
            model, graph_x, graph_ei, graph_ea,
            pert_gene_idx, output_gene_indices,
            H_tensor, n_samples=tta_samples,
        )
    else:
        model.eval()
        with torch.no_grad():
            h_base = model.encode_graph(graph_x, graph_ei, graph_ea)
            w_pred = model.forward_perturbation(
                graph_x, h_base, pert_gene_idx, graph_ei, graph_ea, output_gene_indices
            )
        return w_pred @ H_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta",    type=int, default=config.TTA_SAMPLES)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--output", type=str, default="submission_gnn_v2_local.csv")
    args = parser.parse_args()

    tta_samples = 1 if args.no_tta else args.tta
    config.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, TTA samples: {tta_samples}")

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    W_train, H_np, _ = get_or_build_programs(de_matrix, weight_matrix, gene_order,
                                              K=config.GENE_PROGRAM_K)
    H_tensor = torch.FloatTensor(H_np).to(device)
    K        = H_tensor.shape[0]

    graph_x, graph_ei, graph_ea, gene_to_node, gene_order_out, output_gene_indices = \
        load_graph_to_device(device)

    test_df       = load_pert_ids()
    test_genes    = test_df["pert"].tolist()
    test_pert_ids = test_df["pert_id"].tolist()

    # Load LOCAL fold checkpoints (v2_local_fold{k}_best.pt)
    fold_ckpts = [
        os.path.join(config.CHECKPOINTS_DIR, f"v2_local_fold{k}_best.pt")
        for k in range(1, config.N_FOLDS + 1)
    ]
    available = [p for p in fold_ckpts if os.path.exists(p)]
    if not available:
        raise FileNotFoundError(
            "No local v2 checkpoints found. Run train_v2_local.py first."
        )
    print(f"Loaded {len(available)} local fold checkpoints")

    def load_model(ckpt_path):
        m = GEARSModelV2(
            node_feat_dim=graph_x.shape[1],
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            K=K,
            heads=config.GNN_HEADS,
            dropout=config.GNN_DROPOUT,
        ).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        return m

    models = [load_model(p) for p in available]

    all_preds = []
    for gene in test_genes:
        pert_idx   = gene_to_node.get(gene, 0)
        fold_preds = []

        for model in models:
            de = predict_single_model(
                model, graph_x, graph_ei, graph_ea,
                pert_idx, output_gene_indices,
                H_tensor, tta_samples, device,
            )
            fold_preds.append(de.cpu().numpy())

        avg_pred = np.mean(fold_preds, axis=0)
        all_preds.append(avg_pred)
        print(f"  {gene}: mean|pred|={np.abs(avg_pred).mean():.4f}")

    preds = np.array(all_preds)

    sub = pd.DataFrame(preds, columns=gene_order_out)
    sub.insert(0, "pert_id", test_pert_ids)

    out_path = os.path.join(config.SUBMISSIONS_DIR, args.output)
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  shape={preds.shape}")
    print(f"mean|pred|: {np.abs(preds).mean():.4f}")


if __name__ == "__main__":
    main()
