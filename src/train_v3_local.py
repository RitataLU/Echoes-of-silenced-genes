"""
GNN v3 training — LOCAL (CPU/Mac) variant.

What's new in v3 vs v2
-----------------------
- graph_builder_v3  : adds directed TF regulatory edges (DoRothEA / TRRUST)
- gnn_model_v3      : TransformerConv (dot-product attention) + EdgeEncoder
                      + optional GlobalContextLayer

Checkpoints : v3_local_fold{k}_best.pt  (no overwrite of v2 checkpoints)
OOF output  : oof_gnn_v3_local.csv

Usage
-----
    python src/train_v3_local.py
    python src/train_v3_local.py --k 32 --global-ctx
"""
from __future__ import annotations
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config_local as config
from data_utils import load_ground_truth, load_pert_ids, load_adata, compute_wmae
from graph_builder_v3 import build_graph_v3
from node_features import build_node_features
from gene_programs import get_or_build_programs
from gnn_model_v3 import (
    GEARSModelV3,
    weighted_mae_loss_v3,
    weighted_huber_loss_v3,
    drop_edges,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Graph loading ──────────────────────────────────────────────────────────

def load_graph_to_device(device):
    adata = load_adata()
    node_names, edge_index, edge_attr = build_graph_v3(
        score_threshold=config.STRING_SCORE_GNN, adata=adata
    )
    # edge_attr: (E, 2) — [weight, type]
    node_features = build_node_features(node_names, adata=adata)

    gene_to_node = {g: i for i, g in enumerate(node_names)}
    _, _, gene_order, _ = load_ground_truth()
    output_gene_indices = [gene_to_node[g] for g in gene_order if g in gene_to_node]

    graph_x  = torch.FloatTensor(node_features).to(device)
    graph_ei = torch.LongTensor(edge_index).to(device)
    graph_ea = torch.FloatTensor(edge_attr).to(device)   # (E, 2)

    print(f"\nGraph → {device}:")
    print(f"  Nodes: {graph_x.shape[0]},  feat_dim: {graph_x.shape[1]}")
    print(f"  Edges: {graph_ei.shape[1]},  edge_feat_dim: {graph_ea.shape[1]}")
    print(f"  Output genes: {len(output_gene_indices)}")
    return graph_x, graph_ei, graph_ea, node_names, gene_to_node, output_gene_indices


# ── Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_fold(model, h_base, edge_feats, val_indices,
                  de_matrix, weight_matrix, train_genes, gene_to_node,
                  output_gene_indices, graph_x, graph_ei, graph_ea, H_tensor, device):
    model.eval()
    preds = []
    # Re-encode edges without dropout for eval
    ef = model.encode_edges(graph_ea)
    for i in val_indices:
        gene     = train_genes[i]
        pert_idx = gene_to_node.get(gene, 0)
        w_pred   = model.forward_perturbation(
            graph_x, h_base, pert_idx, graph_ei, ef, output_gene_indices
        )
        de_pred = (w_pred @ H_tensor).cpu().numpy()
        preds.append(de_pred)

    preds = np.array(preds)
    wmae  = compute_wmae(preds, de_matrix[val_indices], weight_matrix[val_indices])
    return wmae, preds


# ── Training loop ─────────────────────────────────────────────────────────

def train_fold(fold, train_indices, val_indices,
               de_matrix, weight_matrix, train_genes,
               W_train, H_tensor,
               graph_x, graph_ei, graph_ea, gene_to_node, output_gene_indices,
               device, K, use_global_ctx, edge_feat_dim):

    model = GEARSModelV3(
        node_feat_dim=graph_x.shape[1],
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        K=K,
        heads=config.GNN_HEADS,
        dropout=config.GNN_DROPOUT,
        edge_feat_dim=edge_feat_dim,
        use_global_ctx=use_global_ctx,
    ).to(device)
    print(f"\n[Fold {fold}] Params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=config.LR_MIN
    )

    best_val   = float("inf")
    patience_c = 0
    history    = {"train_loss": [], "val_wmae": [], "epoch": []}
    ckpt_path  = os.path.join(config.CHECKPOINTS_DIR, f"v3_local_fold{fold}_best.pt")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        indices = train_indices.copy()
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches  = 0

        for batch_start in range(0, len(indices), config.BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start + config.BATCH_SIZE]
            optimizer.zero_grad()

            ei_aug, ea_aug = drop_edges(graph_ei, graph_ea, config.EDGE_DROP_RATE)
            # Pre-encode edge features once per batch (shared across samples)
            edge_feats = model.encode_edges(ea_aug)
            h_base     = model.encode_graph(graph_x, ei_aug, edge_feats)

            total_loss = torch.tensor(0.0, device=device)

            for i in batch_idx:
                gene      = train_genes[i]
                pert_node = gene_to_node.get(gene, 0)

                w_pred = model.forward_perturbation(
                    graph_x, h_base, pert_node, ei_aug, edge_feats, output_gene_indices
                )

                w_true  = torch.FloatTensor(W_train[i]).to(device)
                weights = torch.FloatTensor(weight_matrix[i]).to(device)

                if epoch <= 20:
                    loss = weighted_huber_loss_v3(w_pred, w_true, H_tensor, weights)
                else:
                    loss = weighted_mae_loss_v3(w_pred, w_true, H_tensor, weights)

                total_loss = total_loss + loss

            (total_loss / len(batch_idx)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step(epoch - 1 + batch_start / len(indices))

            epoch_loss += total_loss.item() / len(batch_idx)
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                ef_eval = model.encode_edges(graph_ea)
                h_eval  = model.encode_graph(graph_x, graph_ei, ef_eval)

            val_wmae, _ = evaluate_fold(
                model, h_eval, ef_eval, val_indices,
                de_matrix, weight_matrix, train_genes, gene_to_node,
                output_gene_indices, graph_x, graph_ei, graph_ea, H_tensor, device,
            )

            history["train_loss"].append(avg_loss)
            history["val_wmae"].append(val_wmae)
            history["epoch"].append(epoch)

            improved = val_wmae < best_val
            if improved:
                best_val   = val_wmae
                patience_c = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_c += 1

            marker = " ✓" if improved else ""
            print(f"  Epoch {epoch:4d} | loss: {avg_loss:.4f} | "
                  f"val_wmae: {val_wmae:.4f} | best: {best_val:.4f}{marker}")

            if patience_c >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    history["best_val_wmae"] = best_val
    history["checkpoint"]    = ckpt_path
    print(f"\n[Fold {fold}] Best val WMAE: {best_val:.4f}")
    return history


# ── Plot ──────────────────────────────────────────────────────────────────

def plot_curves(all_history, suffix="v3_local"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, h in enumerate(all_history):
        axes[0].plot(h["epoch"], h["train_loss"], label=f"Fold {i+1}", alpha=0.8)
        axes[1].plot(h["epoch"], h["val_wmae"],   label=f"Fold {i+1}", alpha=0.8)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"Train Loss ({suffix})")
    axes[1].set(xlabel="Epoch", ylabel="Val WMAE", title=f"Val WMAE ({suffix})")
    axes[1].axhline(0.1268, color="gray", linestyle="--", label="Baseline 0.1268")
    for ax in axes:
        ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, f"training_curves_{suffix}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Curves saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",          type=int,  default=config.GENE_PROGRAM_K,
                        help="Number of gene programs (SVD components)")
    parser.add_argument("--edge-feat-dim", type=int, default=32,
                        help="Edge feature projection dimension")
    parser.add_argument("--global-ctx", action="store_true",
                        help="Enable GlobalContextLayer (slower, better long-range)")
    args = parser.parse_args()

    set_seed(config.SEED)
    config.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: hidden={config.GNN_HIDDEN_DIM}, layers={config.GNN_NUM_LAYERS}, "
          f"epochs={config.MAX_EPOCHS}, global_ctx={args.global_ctx}")

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    W_train, H_np, _ = get_or_build_programs(
        de_matrix, weight_matrix, gene_order, K=args.k
    )
    H_tensor = torch.FloatTensor(H_np).to(device)
    K = H_tensor.shape[0]
    print(f"Gene programs: {K}")

    graph_x, graph_ei, graph_ea, node_names, gene_to_node, output_gene_indices = \
        load_graph_to_device(device)

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    all_history = []
    oof_preds   = np.zeros_like(de_matrix)

    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_genes)))):
            train_idx = train_idx.tolist()
            val_idx   = val_idx.tolist()

            print(f"\n{'='*60}")
            print(f" Fold {fold+1}/{config.N_FOLDS} — train: {len(train_idx)}, val: {len(val_idx)}")
            print(f"{'='*60}")

            hist = train_fold(
                fold=fold + 1,
                train_indices=train_idx,
                val_indices=val_idx,
                de_matrix=de_matrix,
                weight_matrix=weight_matrix,
                train_genes=train_genes,
                W_train=W_train,
                H_tensor=H_tensor,
                graph_x=graph_x,
                graph_ei=graph_ei,
                graph_ea=graph_ea,
                gene_to_node=gene_to_node,
                output_gene_indices=output_gene_indices,
                device=device,
                K=K,
                use_global_ctx=args.global_ctx,
                edge_feat_dim=args.edge_feat_dim,
            )
            all_history.append(hist)

            # OOF predictions from best checkpoint
            best_model = GEARSModelV3(
                node_feat_dim=graph_x.shape[1],
                hidden_dim=config.GNN_HIDDEN_DIM,
                num_layers=config.GNN_NUM_LAYERS,
                K=K,
                heads=config.GNN_HEADS,
                dropout=config.GNN_DROPOUT,
                edge_feat_dim=args.edge_feat_dim,
                use_global_ctx=args.global_ctx,
            ).to(device)
            best_model.load_state_dict(
                torch.load(hist["checkpoint"], map_location=device)
            )
            best_model.eval()

            with torch.no_grad():
                ef_eval = best_model.encode_edges(graph_ea)
                h_eval  = best_model.encode_graph(graph_x, graph_ei, ef_eval)

            _, oof_fold = evaluate_fold(
                best_model, h_eval, ef_eval, val_idx,
                de_matrix, weight_matrix, train_genes, gene_to_node,
                output_gene_indices, graph_x, graph_ei, graph_ea, H_tensor, device,
            )
            oof_preds[val_idx] = oof_fold

        oof_wmae = compute_wmae(oof_preds, de_matrix, weight_matrix)
        print(f"\n{'='*60}")
        print(f" OOF WMAE:      {oof_wmae:.4f}")
        print(f" Baseline:      0.1268")
        print(f" Improvement:   {(0.1268 - oof_wmae) / 0.1268 * 100:+.1f}%")
        print(f" vs GNN v2:     {(0.1167 - oof_wmae) / 0.1167 * 100:+.1f}%")
        print(f"{'='*60}")

        oof_df = pd.DataFrame(oof_preds, columns=gene_order)
        oof_df.insert(0, "pert_id", train_genes)
        oof_df.to_csv(
            os.path.join(config.SUBMISSIONS_DIR, "oof_gnn_v3_local.csv"), index=False
        )

    finally:
        if all_history:
            plot_curves(all_history, suffix="v3_local")


if __name__ == "__main__":
    main()
