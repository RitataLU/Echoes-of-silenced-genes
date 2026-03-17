"""
Generate submission CSV from trained GNN fold checkpoints.

Loads each fold's best model, averages predictions across folds (ensemble),
and writes outputs/submissions/submission_gnn.csv.

Usage:
    python src/predict.py
    python src/predict.py --folds 1 2 3   # use only specific folds
"""
from __future__ import annotations
import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, load_pert_ids, compute_wmae
from graph_builder import build_graph
from node_features import build_node_features
from gnn_model import GEARSModel
from train import load_graph_to_device


@torch.no_grad()
def predict_all_perts(model: GEARSModel,
                      h_base: torch.Tensor,
                      test_genes: List[str],
                      gene_to_node_idx: dict,
                      output_gene_indices: List[int],
                      graph_ei: torch.Tensor,
                      graph_ea: torch.Tensor,
                      device: torch.device) -> np.ndarray:
    """
    Predict DE for a list of test genes.

    Returns
    -------
    preds : (n_test, n_genes) float32
    """
    model.eval()
    preds = []
    for gene in test_genes:
        pert_idx = gene_to_node_idx.get(gene)
        if pert_idx is None:
            print(f"  WARNING: gene '{gene}' not in graph — predicting zeros")
            preds.append(np.zeros(len(output_gene_indices), dtype=np.float32))
            continue

        de_pred = model.forward_perturbation(
            h_base, pert_idx, graph_ei, graph_ea, output_gene_indices
        )
        preds.append(de_pred.cpu().numpy())

    return np.array(preds, dtype=np.float32)


def generate_submission(fold_ids: Optional[List[int]] = None,
                         output_name: str = "submission_gnn.csv",
                         allow_knn_fallback: bool = True) -> str:
    """
    Load fold checkpoints, ensemble predictions, save submission.

    Parameters
    ----------
    fold_ids   : list of fold numbers to use (1-indexed). None = use all available.
    output_name: filename for the submission CSV.

    Returns
    -------
    path to saved submission CSV
    """
    config.make_dirs()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # MPS disabled: PyG GATConv scatter ops abort on MPS with large graphs
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Discover available checkpoints
    available_checkpoints = {}
    for fold in range(1, config.N_FOLDS + 1):
        ckpt = os.path.join(config.CHECKPOINTS_DIR, f"fold{fold}_best.pt")
        if os.path.exists(ckpt):
            available_checkpoints[fold] = ckpt

    if not available_checkpoints:
        if allow_knn_fallback:
            knn_path = os.path.join(config.SUBMISSIONS_DIR, "submission_knn.csv")
            if os.path.exists(knn_path):
                out_path = os.path.join(config.SUBMISSIONS_DIR, output_name)
                if os.path.abspath(knn_path) != os.path.abspath(out_path):
                    shutil.copyfile(knn_path, out_path)
                print(
                    "WARNING: No GNN checkpoints found. "
                    f"Using KNN fallback from {knn_path}."
                )
                print(f"Submission saved to: {out_path}")
                return out_path

        raise FileNotFoundError(
            f"No checkpoints found in {config.CHECKPOINTS_DIR}. "
            "Run train.py first, or run src/knn_baseline.py to create "
            "outputs/submissions/submission_knn.csv and use fallback."
        )

    if fold_ids is not None:
        available_checkpoints = {k: v for k, v in available_checkpoints.items()
                                  if k in fold_ids}
        if not available_checkpoints:
            raise ValueError(
                "Requested folds are unavailable. "
                f"Available folds: {sorted([f for f in range(1, config.N_FOLDS + 1) if os.path.exists(os.path.join(config.CHECKPOINTS_DIR, f'fold{f}_best.pt'))])}."
            )

    print(f"Using {len(available_checkpoints)} fold(s): {sorted(available_checkpoints)}")

    # Load graph (once)
    _, _, gene_order, _ = load_ground_truth()
    graph_x, graph_ei, graph_ea, node_names, gene_to_node_idx, output_gene_indices = \
        load_graph_to_device(device)

    # Test perturbations
    perts_df = load_pert_ids()
    test_genes   = perts_df["pert"].tolist()
    pert_ids     = perts_df["pert_id"].tolist()

    print(f"Predicting for {len(test_genes)} test perturbations …")

    # Collect predictions from each fold
    all_fold_preds = []
    for fold, ckpt_path in sorted(available_checkpoints.items()):
        print(f"  Loading fold {fold}: {ckpt_path}")
        model = GEARSModel(
            node_feat_dim=config.NODE_FEAT_DIM,
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            output_dim=len(output_gene_indices),
            heads=config.GNN_HEADS,
            dropout=config.GNN_DROPOUT,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        h_base = model.encode_graph(graph_x, graph_ei, graph_ea)
        fold_preds = predict_all_perts(
            model, h_base, test_genes, gene_to_node_idx,
            output_gene_indices, graph_ei, graph_ea, device,
        )
        all_fold_preds.append(fold_preds)
        print(f"    Fold {fold} predictions: {fold_preds.shape}, "
              f"mean |pred|: {np.abs(fold_preds).mean():.4f}")

    # Average ensemble
    final_preds = np.mean(all_fold_preds, axis=0)  # (n_test, n_genes)
    print(f"\nEnsemble predictions: {final_preds.shape}")
    print(f"  mean |pred|: {np.abs(final_preds).mean():.4f}")

    # Build submission DataFrame
    sub = pd.DataFrame(final_preds, columns=gene_order)
    sub.insert(0, "pert_id", pert_ids)

    out_path = os.path.join(config.SUBMISSIONS_DIR, output_name)
    sub.to_csv(out_path, index=False)
    print(f"\nGNN submission saved to: {out_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Specific fold numbers to use (default: all)")
    parser.add_argument("--output", type=str, default="submission_gnn.csv")
    parser.add_argument("--no-knn-fallback", action="store_true",
                        help="Disable fallback to submission_knn.csv when no checkpoints are found")
    args = parser.parse_args()

    path = generate_submission(
        fold_ids=args.folds,
        output_name=args.output,
        allow_knn_fallback=not args.no_knn_fallback,
    )
    print(f"\nDone! Submit: {path}")
