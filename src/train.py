"""
GNN training script with 5-fold cross-validation.

Usage:
    python src/train.py

Each fold trains a separate GEARSModel and saves the best checkpoint to
outputs/checkpoints/fold{k}_best.pt. Training curves are saved to
outputs/figures/.

Key design decisions:
- Graph (x, edge_index, edge_attr) is kept as a static GPU object
- encode_graph() is called once per epoch (not cached between epochs)
- Weighted MAE loss matches the competition metric exactly
- DropEdge + label noise as data augmentation
"""
from __future__ import annotations
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import (
    load_ground_truth, load_pert_ids, load_adata,
    compute_wmae, compute_wmae_per_pert
)
from graph_builder import build_graph
from node_features import build_node_features
from gnn_model import GEARSModel, weighted_mae_loss, weighted_huber_loss, drop_edges


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Graph loading ──────────────────────────────────────────────────────────

def load_graph_to_device(device: torch.device):
    """
    Build or load the gene interaction graph and move to device.

    Returns
    -------
    graph_x          : (n_nodes, node_feat_dim) tensor on device
    graph_edge_index : (2, n_edges) tensor on device
    graph_edge_attr  : (n_edges,) tensor on device
    node_names       : list of gene names
    gene_to_node_idx : dict mapping gene name → node index
    output_gene_indices: list of indices for 5127 output genes
    """
    adata = load_adata()

    node_names, edge_index, edge_attr = build_graph(
        score_threshold=config.STRING_SCORE_GNN,
        adata=adata,
    )
    node_features = build_node_features(node_names, adata=adata)

    gene_to_node_idx = {g: i for i, g in enumerate(node_names)}

    # Output gene indices (the 5127 scored genes in submission order)
    _, _, gene_order, _ = load_ground_truth()
    output_gene_indices = [gene_to_node_idx[g] for g in gene_order
                           if g in gene_to_node_idx]
    missing = [g for g in gene_order if g not in gene_to_node_idx]
    if missing:
        print(f"WARNING: {len(missing)} output genes not in graph: {missing[:5]} …")

    graph_x = torch.FloatTensor(node_features).to(device)
    graph_ei = torch.LongTensor(edge_index).to(device)
    graph_ea = torch.FloatTensor(edge_attr).to(device)

    print(f"\nGraph on {device}:")
    print(f"  Nodes: {graph_x.shape[0]}, node_feat_dim: {graph_x.shape[1]}")
    print(f"  Edges: {graph_ei.shape[1]}")
    print(f"  Output genes: {len(output_gene_indices)}")

    return graph_x, graph_ei, graph_ea, node_names, gene_to_node_idx, output_gene_indices


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_fold(model: GEARSModel,
                  h_base: torch.Tensor,
                  val_indices: List[int],
                  de_matrix: np.ndarray,
                  weight_matrix: np.ndarray,
                  train_genes: List[str],
                  gene_to_node_idx: dict,
                  output_gene_indices: List[int],
                  graph_ei: torch.Tensor,
                  graph_ea: torch.Tensor,
                  device: torch.device) -> Tuple[float, np.ndarray]:
    """
    Evaluate model on a validation fold.

    Returns
    -------
    wmae  : scalar weighted MAE
    preds : (n_val, n_genes) predictions
    """
    model.eval()
    preds = []
    for i in val_indices:
        gene = train_genes[i]
        pert_idx = gene_to_node_idx.get(gene, 0)
        de_pred = model.forward_perturbation(
            h_base, pert_idx, graph_ei, graph_ea, output_gene_indices
        )
        preds.append(de_pred.cpu().numpy())

    preds = np.array(preds)
    gt    = de_matrix[val_indices]
    w     = weight_matrix[val_indices]
    wmae  = compute_wmae(preds, gt, w)
    return wmae, preds


# ── Training loop ──────────────────────────────────────────────────────────

def train_fold(fold: int,
               train_indices: List[int],
               val_indices: List[int],
               de_matrix: np.ndarray,
               weight_matrix: np.ndarray,
               train_genes: List[str],
               graph_x: torch.Tensor,
               graph_ei: torch.Tensor,
               graph_ea: torch.Tensor,
               gene_to_node_idx: dict,
               output_gene_indices: List[int],
               device: torch.device) -> dict:
    """Train one fold. Returns training history dict."""

    model = GEARSModel(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        output_dim=len(output_gene_indices),
        heads=config.GNN_HEADS,
        dropout=config.GNN_DROPOUT,
    ).to(device)
    print(f"\n[Fold {fold}] Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.MAX_EPOCHS,
        eta_min=config.LR_MIN,
    )

    best_val_wmae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_wmae": [], "epoch": []}

    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"fold{fold}_best.pt")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()

        # Shuffle training indices
        indices = train_indices.copy()
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(indices), config.BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start + config.BATCH_SIZE]
            optimizer.zero_grad()

            # Apply DropEdge and encode graph fresh per batch so that
            # backward() can free the graph after each optimizer step.
            ei_aug, ea_aug = drop_edges(graph_ei, graph_ea, config.EDGE_DROP_RATE)
            h_base = model.encode_graph(graph_x, ei_aug, ea_aug)

            total_loss = torch.tensor(0.0, device=device)

            for i in batch_idx:
                gene = train_genes[i]
                pert_node = gene_to_node_idx.get(gene, 0)

                de_pred = model.forward_perturbation(
                    h_base, pert_node, ei_aug, ea_aug, output_gene_indices
                )

                de_gt = torch.FloatTensor(de_matrix[i]).to(device)
                w     = torch.FloatTensor(weight_matrix[i]).to(device)

                # Label noise
                if config.LABEL_NOISE_STD > 0:
                    de_gt = de_gt + torch.randn_like(de_gt) * config.LABEL_NOISE_STD

                # Use Huber for first 50 epochs, then switch to MAE
                if epoch <= 50:
                    loss = weighted_huber_loss(de_pred, de_gt, w)
                else:
                    loss = weighted_mae_loss(de_pred, de_gt, w)

                total_loss = total_loss + loss

            (total_loss / len(batch_idx)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += total_loss.item() / len(batch_idx)
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation (every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            # Re-encode without DropEdge for fair evaluation
            with torch.no_grad():
                h_base_eval = model.encode_graph(graph_x, graph_ei, graph_ea)

            val_wmae, _ = evaluate_fold(
                model, h_base_eval, val_indices,
                de_matrix, weight_matrix, train_genes,
                gene_to_node_idx, output_gene_indices,
                graph_ei, graph_ea, device,
            )

            history["train_loss"].append(avg_train_loss)
            history["val_wmae"].append(val_wmae)
            history["epoch"].append(epoch)

            improved = val_wmae < best_val_wmae
            if improved:
                best_val_wmae = val_wmae
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1

            marker = " ✓" if improved else ""
            print(f"  Epoch {epoch:4d} | train_loss: {avg_train_loss:.4f} | "
                  f"val_wmae: {val_wmae:.4f} | best: {best_val_wmae:.4f}{marker}")

            if patience_counter >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={config.PATIENCE})")
                break

    history["best_val_wmae"] = best_val_wmae
    history["checkpoint"]    = checkpoint_path
    print(f"\n[Fold {fold}] Best val WMAE: {best_val_wmae:.4f}")
    return history


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_training_curves(all_history: List[dict], save: bool = True, show: bool = False):
    """Plot training curves for all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold_i, hist in enumerate(all_history):
        axes[0].plot(hist["epoch"], hist["train_loss"],
                     label=f"Fold {fold_i+1}", alpha=0.8)
        axes[1].plot(hist["epoch"], hist["val_wmae"],
                     label=f"Fold {fold_i+1}", alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted MAE loss")
    axes[0].set_title("Training Loss (per fold)")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val WMAE")
    axes[1].set_title("Validation WMAE (per fold)")
    axes[1].axhline(0.1268, color="gray", linestyle="--", linewidth=1, label="Baseline 0.1268")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(config.FIGURES_DIR, "training_curves.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Training curves saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    set_seed(config.SEED)
    config.make_dirs()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    # MPS disabled: PyG GATConv scatter ops abort on MPS with large graphs
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    # Load graph
    graph_x, graph_ei, graph_ea, node_names, gene_to_node_idx, output_gene_indices = \
        load_graph_to_device(device)

    # 5-fold CV
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    all_history = []
    oof_preds   = np.zeros_like(de_matrix)

    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_genes)))):
            train_idx = train_idx.tolist()
            val_idx   = val_idx.tolist()

            print(f"\n{'='*60}")
            print(f" Fold {fold+1}/{config.N_FOLDS} — "
                  f"train: {len(train_idx)}, val: {len(val_idx)}")
            print(f"{'='*60}")

            hist = train_fold(
                fold=fold + 1,
                train_indices=train_idx,
                val_indices=val_idx,
                de_matrix=de_matrix,
                weight_matrix=weight_matrix,
                train_genes=train_genes,
                graph_x=graph_x,
                graph_ei=graph_ei,
                graph_ea=graph_ea,
                gene_to_node_idx=gene_to_node_idx,
                output_gene_indices=output_gene_indices,
                device=device,
            )
            all_history.append(hist)

            # Collect OOF predictions
            best_model = GEARSModel(
                node_feat_dim=config.NODE_FEAT_DIM,
                hidden_dim=config.GNN_HIDDEN_DIM,
                num_layers=config.GNN_NUM_LAYERS,
                output_dim=len(output_gene_indices),
                heads=config.GNN_HEADS,
                dropout=config.GNN_DROPOUT,
            ).to(device)
            best_model.load_state_dict(torch.load(hist["checkpoint"], map_location=device))
            best_model.eval()

            with torch.no_grad():
                h_base_eval = best_model.encode_graph(graph_x, graph_ei, graph_ea)

            _, oof_fold_preds = evaluate_fold(
                best_model, h_base_eval, val_idx,
                de_matrix, weight_matrix, train_genes,
                gene_to_node_idx, output_gene_indices,
                graph_ei, graph_ea, device,
            )
            oof_preds[val_idx] = oof_fold_preds

        # Overall OOF WMAE
        oof_wmae = compute_wmae(oof_preds, de_matrix, weight_matrix)
        print(f"\n{'='*60}")
        print(f" Overall OOF WMAE: {oof_wmae:.4f}")
        print(f" Baseline WMAE:    0.1268")
        print(f" Improvement:      {(0.1268 - oof_wmae) / 0.1268 * 100:+.1f}%")
        print(f"{'='*60}")

        # Per-fold summary
        print("\nPer-fold best val WMAE:")
        for i, h in enumerate(all_history):
            print(f"  Fold {i+1}: {h['best_val_wmae']:.4f}")

        # Save OOF predictions
        oof_df = pd.DataFrame(oof_preds, columns=gene_order)
        oof_df.insert(0, "pert_id", train_genes)
        oof_path = os.path.join(config.SUBMISSIONS_DIR, "oof_gnn.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"\nOOF predictions saved to: {oof_path}")
    finally:
        if all_history:
            print("\nSaving training curves from available fold history …")
            plot_training_curves(all_history, save=True, show=False)
        else:
            print("\nNo fold history available; skipping training curves plot.")


if __name__ == "__main__":
    main()
