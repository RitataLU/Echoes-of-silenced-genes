"""
GNN v3 inference — LOCAL (CPU/Mac) variant.

Loads v3_local_fold{k}_best.pt checkpoints from train_v3_local.py.
Output: submission_gnn_v3_local.csv

Usage
-----
    python src/predict_v3_local.py
    python src/predict_v3_local.py --tta 10
    python src/predict_v3_local.py --no-tta
    python src/predict_v3_local.py --global-ctx   # if trained with --global-ctx
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
from data_utils import load_ground_truth, load_pert_ids, load_adata
from graph_builder_v3 import build_graph_v3
from node_features import build_node_features
from gene_programs import get_or_build_programs
from gnn_model_v3 import GEARSModelV3, mc_dropout_predict_v3


def load_graph_to_device(device):
    adata = load_adata()
    node_names, edge_index, edge_attr = build_graph_v3(
        score_threshold=config.STRING_SCORE_GNN, adata=adata
    )
    node_features = build_node_features(node_names, adata=adata)
    gene_to_node  = {g: i for i, g in enumerate(node_names)}

    _, _, gene_order, _ = load_ground_truth()
    output_gene_indices = [gene_to_node[g] for g in gene_order if g in gene_to_node]

    graph_x  = torch.FloatTensor(node_features).to(device)
    graph_ei = torch.LongTensor(edge_index).to(device)
    graph_ea = torch.FloatTensor(edge_attr).to(device)   # (E, 2)
    return graph_x, graph_ei, graph_ea, gene_to_node, gene_order, output_gene_indices


def predict_single_model(model, graph_x, graph_ei, graph_ea,
                         pert_gene_idx, output_gene_indices,
                         H_tensor, tta_samples, device):
    if tta_samples > 1:
        return mc_dropout_predict_v3(
            model, graph_x, graph_ei, graph_ea,
            pert_gene_idx, output_gene_indices,
            H_tensor, n_samples=tta_samples,
        )
    else:
        model.eval()
        with torch.no_grad():
            edge_feats = model.encode_edges(graph_ea)
            h_base     = model.encode_graph(graph_x, graph_ei, edge_feats)
            w_pred     = model.forward_perturbation(
                graph_x, h_base, pert_gene_idx, graph_ei, edge_feats, output_gene_indices
            )
        return w_pred @ H_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",            type=int,  default=config.GENE_PROGRAM_K,
                        help="Must match the --k used during training")
    parser.add_argument("--tta",          type=int,  default=config.TTA_SAMPLES)
    parser.add_argument("--no-tta",       action="store_true")
    parser.add_argument("--edge-feat-dim", type=int, default=32)
    parser.add_argument("--global-ctx",   action="store_true")
    parser.add_argument("--output",       type=str,  default="submission_gnn_v3_local.csv")
    args = parser.parse_args()

    tta_samples = 1 if args.no_tta else args.tta
    config.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, TTA samples: {tta_samples}")

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    W_train, H_np, _ = get_or_build_programs(
        de_matrix, weight_matrix, gene_order, K=args.k
    )
    H_tensor = torch.FloatTensor(H_np).to(device)
    K        = H_tensor.shape[0]

    graph_x, graph_ei, graph_ea, gene_to_node, gene_order_out, output_gene_indices = \
        load_graph_to_device(device)

    # Load v3 local fold checkpoints
    fold_ckpts = [
        os.path.join(config.CHECKPOINTS_DIR, f"v3_local_fold{k}_best.pt")
        for k in range(1, config.N_FOLDS + 1)
    ]
    available = [p for p in fold_ckpts if os.path.exists(p)]
    if not available:
        raise FileNotFoundError(
            "No v3 local checkpoints found. Run train_v3_local.py first."
        )
    print(f"Loaded {len(available)} v3 fold checkpoints")

    def load_model(ckpt_path):
        m = GEARSModelV3(
            node_feat_dim=graph_x.shape[1],
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            K=K,
            heads=config.GNN_HEADS,
            dropout=config.GNN_DROPOUT,
            edge_feat_dim=args.edge_feat_dim,
            use_global_ctx=args.global_ctx,
        ).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        return m

    models = [load_model(p) for p in available]

    test_df       = load_pert_ids()
    test_genes    = test_df["pert"].tolist()
    test_pert_ids = test_df["pert_id"].tolist()

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
