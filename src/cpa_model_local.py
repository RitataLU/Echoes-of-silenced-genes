"""
CPA (Compositional Perturbation Autoencoder) — LOCAL (CPU/Mac) variant.

Identical to cpa_model.py except:
- Uses config_local.py (smaller model, fewer epochs, CPU-friendly)
- Checkpoint: cpa_local_best.pt  (no overwrite of GPU checkpoint)
- Output:     submission_cpa_local.csv

Usage
-----
    python src/cpa_model_local.py            # train + predict
    python src/cpa_model_local.py --predict  # predict only (needs checkpoint)
"""
from __future__ import annotations
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
import config_local as config
from data_utils import load_ground_truth, load_pert_ids, load_adata, compute_wmae
from graph_builder import build_graph
from node_features import build_node_features
# Reuse the CPA class and data utilities from cpa_model.py
from cpa_model import CPA, prepare_cell_data, reconstruction_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training (local config) ───────────────────────────────────────────────

def train_cpa_local(n_epochs: int = None,
                    latent_dim: int = None,
                    lr: float = 1e-3,
                    batch_size: int = None,
                    device_str: str = "cpu"):

    # Defaults from local config
    if n_epochs   is None: n_epochs   = config.CPA_EPOCHS
    if latent_dim is None: latent_dim = config.CPA_LATENT_DIM
    if batch_size is None: batch_size = config.CPA_BATCH_SIZE
    hidden_dims = config.CPA_HIDDEN_DIMS

    set_seed(config.SEED)
    config.make_dirs()

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"CPA-local training on {device}  "
          f"(latent={latent_dim}, hidden={hidden_dims}, epochs={n_epochs})")

    # ── Load data ──────────────────────────────────────────────────────
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    adata = load_adata()

    ctrl_expr, pert_data, gene_indices = prepare_cell_data(adata, gene_order, device)
    n_genes = ctrl_expr.shape[1]

    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_features = build_node_features(node_names, adata=adata)
    gene_to_node  = {g: i for i, g in enumerate(node_names)}
    feat_dim      = node_features.shape[1]

    def get_feat(gene):
        idx = gene_to_node.get(gene, 0)
        return torch.FloatTensor(node_features[idx]).to(device)

    gene_weights = torch.FloatTensor(weight_matrix.mean(axis=0)).to(device)

    # ── Model ──────────────────────────────────────────────────────────
    model = CPA(
        n_genes=n_genes,
        latent_dim=latent_dim,
        gene_feat_dim=feat_dim,
        hidden_dims=hidden_dims,
        dropout=0.1,
    ).to(device)
    print(f"CPA-local parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )

    best_wmae  = float("inf")
    ckpt_path  = os.path.join(config.CHECKPOINTS_DIR, "cpa_local_best.pt")
    train_perts = [g for g in train_genes if g in pert_data]

    print(f"\nTraining CPA-local for {n_epochs} epochs …")
    print(f"  Training perturbations with cell data: {len(train_perts)} / {len(train_genes)}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        random.shuffle(train_perts)
        epoch_loss = 0.0

        for pert_gene in train_perts:
            gene_feat  = get_feat(pert_gene)
            pert_cells = pert_data[pert_gene]
            n_cells    = pert_cells.shape[0]

            ctrl_idx  = torch.randint(0, ctrl_expr.shape[0], (batch_size,), device=device)
            pert_idx_ = torch.randint(0, n_cells,            (batch_size,), device=device)
            x_ctrl_b  = ctrl_expr[ctrl_idx]
            x_pert_b  = pert_cells[pert_idx_]

            optimizer.zero_grad()

            x_recon, _, delta = model(x_ctrl_b, gene_feat)
            loss_recon  = reconstruction_loss(x_recon, x_pert_b, weights=gene_weights)
            loss_delta  = 0.01 * torch.exp(-delta.pow(2).mean())
            loss        = loss_recon + loss_delta
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_perts), 1)

        eval_every = max(1, n_epochs // 10)
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            preds = []
            for pert_gene in train_genes:
                gene_feat = get_feat(pert_gene)
                de_pred   = model.predict_de(ctrl_expr, gene_feat).cpu().numpy()
                preds.append(de_pred)

            preds    = np.array(preds)
            oof_wmae = compute_wmae(preds, de_matrix, weight_matrix)

            improved = oof_wmae < best_wmae
            if improved:
                best_wmae = oof_wmae
                torch.save(model.state_dict(), ckpt_path)
            marker = " ✓" if improved else ""
            print(f"  Epoch {epoch:4d} | loss: {avg_loss:.4f} | "
                  f"train_wmae: {oof_wmae:.4f} | best: {best_wmae:.4f}{marker}")

    print(f"\nBest train WMAE: {best_wmae:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path


# ── Prediction ────────────────────────────────────────────────────────────

def predict_cpa_local(ckpt_path: str | None = None,
                      device_str: str = "cpu") -> str:
    config.make_dirs()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    adata = load_adata()
    ctrl_expr, _, gene_indices = prepare_cell_data(adata, gene_order, device)
    n_genes = ctrl_expr.shape[1]

    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_features = build_node_features(node_names, adata=adata)
    gene_to_node  = {g: i for i, g in enumerate(node_names)}
    feat_dim      = node_features.shape[1]

    model = CPA(
        n_genes=n_genes,
        latent_dim=config.CPA_LATENT_DIM,
        gene_feat_dim=feat_dim,
        hidden_dims=config.CPA_HIDDEN_DIMS,
        dropout=0.1,
    ).to(device)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "cpa_local_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_df    = load_pert_ids()
    test_genes = test_df["pert"].tolist()
    test_ids   = test_df["pert_id"].tolist()

    preds = []
    for gene in test_genes:
        idx       = gene_to_node.get(gene, 0)
        gene_feat = torch.FloatTensor(node_features[idx]).to(device)
        de_pred   = model.predict_de(ctrl_expr, gene_feat).cpu().numpy()
        preds.append(de_pred)
        print(f"  {gene}: mean|DE|={np.abs(de_pred).mean():.4f}")

    preds = np.array(preds)

    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)
    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_cpa_local.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nCPA-local submission saved: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict",    action="store_true")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--latent-dim", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=None)
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if args.predict:
        predict_cpa_local(device_str=device_str)
    else:
        ckpt = train_cpa_local(
            n_epochs=args.epochs,
            latent_dim=args.latent_dim,
            lr=args.lr,
            batch_size=args.batch_size,
            device_str=device_str,
        )
        predict_cpa_local(ckpt_path=ckpt, device_str=device_str)
