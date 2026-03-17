"""
CPA — Compositional Perturbation Autoencoder.

Why this is fundamentally different from GNN/KNN
-------------------------------------------------
KNN and GNN work on the AGGREGATED mean DE matrix (80 × 5127).
They throw away all cell-level information: variance, bimodality,
cell-type heterogeneity, and most importantly the DISTRIBUTION of
expression changes.

CPA works directly on INDIVIDUAL CELLS:
    - Encoder maps each cell's expression to a latent vector z
    - z = z_basal + δ_pert  (additive perturbation effect in latent space)
    - Decoder reconstructs expression from z
    - For zero-shot: δ_pert = MLP(gene_features) — generalises to unseen genes

Architecture
------------
                        x_cell (n_expressed_genes)
                              ↓
                    Encoder (MLP or Transformer)
                              ↓
               z_basal  (latent dim, 64)  ← basal cell state
                    +
               δ_pert   = MLP(gene_features[pert_gene])
                              ↓
                    Decoder (MLP)
                              ↓
                   x_reconstructed

Training signal: reconstruction loss on perturbed cells vs control cells.
The model must learn δ_pert such that z_basal + δ_pert decodes to the
perturbed expression profile.

Zero-shot prediction
--------------------
For an unseen test gene:
    1. Compute δ_pert_new = MLP(gene_features[test_gene])
    2. Sample z_basal from control cells (or use the mean)
    3. Decode(z_basal + δ_pert_new) → predicted perturbed expression
    4. DE = mean(predicted_perturbed) - mean(control)

This is the core idea from the original CPA paper (Lotfollahi et al. 2021):
    https://www.science.org/doi/10.1126/science.abc6544

Simplifications here
--------------------
- We use a deterministic AE (not VAE) for stability with 17K cells
- Reconstruction is on the 5127 scored genes only (not all 19226)
- Perturbation embedding is conditioned on node features (same as GNN)
- A single drug/pert adversarial classifier is NOT included here
  (can be added as an extension if needed)

Usage
-----
    python src/cpa_model.py            # train
    python src/cpa_model.py --predict  # generate submission after training
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
import config
from data_utils import load_ground_truth, load_pert_ids, load_adata, compute_wmae
from graph_builder import build_graph
from node_features import build_node_features


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data preparation ──────────────────────────────────────────────────────

def prepare_cell_data(adata, gene_order: list[str], device):
    """
    Extract cell expression matrices for:
        - control cells (non-targeting)
        - each training perturbation condition

    Returns
    -------
    ctrl_expr   : (n_ctrl, n_scored_genes) float32 tensor
    pert_data   : dict {pert_gene: (n_cells, n_scored_genes) tensor}
    gene_indices: indices into adata.var for the 5127 scored genes
    """
    import scipy.sparse as sp

    gene_names   = adata.var_names.tolist()
    gene_to_col  = {g: i for i, g in enumerate(gene_names)}
    gene_indices = [gene_to_col[g] for g in gene_order if g in gene_to_col]
    present_genes = [g for g in gene_order if g in gene_to_col]

    def extract(mask):
        X = adata[mask].X
        if sp.issparse(X):
            X = X.toarray()
        return X[:, gene_indices].astype(np.float32)

    ctrl_mask = adata.obs["sgrna_symbol"] == "non-targeting"
    ctrl_expr = torch.FloatTensor(extract(ctrl_mask)).to(device)

    pert_data = {}
    for pert_gene in adata.obs["sgrna_symbol"].unique():
        if pert_gene == "non-targeting":
            continue
        mask = adata.obs["sgrna_symbol"] == pert_gene
        if mask.sum() < 5:
            continue
        pert_data[pert_gene] = torch.FloatTensor(extract(mask)).to(device)

    print(f"Control cells: {ctrl_expr.shape[0]}")
    print(f"Training perturbations with cells: {len(pert_data)}")
    print(f"Scored genes found in h5ad: {len(present_genes)} / {len(gene_order)}")
    return ctrl_expr, pert_data, gene_indices


# ── Model ─────────────────────────────────────────────────────────────────

class CPA(nn.Module):
    """
    Compositional Perturbation Autoencoder.

    Parameters
    ----------
    n_genes    : number of input/output genes (5127)
    latent_dim : latent space dimension
    gene_feat_dim : gene feature vector dimension (for δ_pert MLP)
    hidden_dims   : hidden layer sizes for encoder/decoder
    dropout       : dropout rate
    """

    def __init__(self,
                 n_genes: int = 5127,
                 latent_dim: int = 64,
                 gene_feat_dim: int = 119,
                 hidden_dims: list[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.latent_dim = latent_dim

        # ── Encoder: cell expression → basal latent ───────────────────
        enc_layers = []
        in_dim = n_genes
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Perturbation embedding: gene features → δ_pert ────────────
        # This is the zero-shot-capable part: δ_pert = MLP(gene_embedding)
        self.pert_mlp = nn.Sequential(
            nn.Linear(gene_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Tanh(),   # bound the perturbation magnitude
        )

        # ── Decoder: latent → reconstructed expression ────────────────
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_genes))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        """x: (batch, n_genes) → z_basal: (batch, latent_dim)"""
        return self.encoder(x)

    def perturb_embed(self, gene_feat):
        """gene_feat: (gene_feat_dim,) or (batch, gene_feat_dim) → δ: same leading dims × latent_dim"""
        return self.pert_mlp(gene_feat)

    def decode(self, z):
        """z: (batch, latent_dim) → x_reconstructed: (batch, n_genes)"""
        return self.decoder(z)

    def forward(self, x_ctrl, gene_feat_pert):
        """
        x_ctrl      : (batch, n_genes) — control cell expressions
        gene_feat_pert: (gene_feat_dim,) — feature of the perturbed gene

        Returns
        -------
        x_recon : (batch, n_genes) — predicted perturbed expression
        z_basal : (batch, latent_dim)
        delta   : (latent_dim,)
        """
        z_basal = self.encode(x_ctrl)                         # (B, L)
        delta   = self.perturb_embed(gene_feat_pert)          # (L,)
        z_pert  = z_basal + delta.unsqueeze(0)                # (B, L)
        x_recon = self.decode(z_pert)                         # (B, G)
        return x_recon, z_basal, delta

    def predict_de(self, ctrl_expr, gene_feat_pert):
        """
        Predict DE vector from control cells + perturbation gene features.

        de_pred = mean(x_reconstructed) - mean(x_ctrl_recon)

        Returns (n_genes,)
        """
        with torch.no_grad():
            x_recon, z_basal, _ = self.forward(ctrl_expr, gene_feat_pert)
            # Reconstruct control without perturbation to get clean baseline
            x_ctrl_recon = self.decode(z_basal)
            de_pred = x_recon.mean(0) - x_ctrl_recon.mean(0)
        return de_pred


# ── Loss functions ────────────────────────────────────────────────────────

def reconstruction_loss(x_recon, x_target, weights=None):
    """Weighted MAE reconstruction loss."""
    diff = torch.abs(x_recon - x_target)
    if weights is not None:
        diff = diff * weights
    return diff.mean()


def contrastive_loss(z_pert_a, z_pert_b, z_ctrl, margin=1.0):
    """
    Encourage perturbation embeddings to be distinct from control.
    Push δ_a and δ_b apart if they are different perturbations.
    (Optional regularisation — comment out if unstable)
    """
    dist_pos = F.mse_loss(z_pert_a, z_pert_b)
    dist_neg_a = F.mse_loss(z_pert_a, z_ctrl)
    dist_neg_b = F.mse_loss(z_pert_b, z_ctrl)
    return torch.clamp(margin - dist_neg_a, min=0) + torch.clamp(margin - dist_neg_b, min=0)


# ── Training ──────────────────────────────────────────────────────────────

def train_cpa(n_epochs: int = 200,
              latent_dim: int = 64,
              lr: float = 1e-3,
              batch_size: int = 256,
              device_str: str = "cuda"):
    set_seed(config.SEED)
    config.make_dirs()

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"CPA training on {device}")

    # ── Load data ──────────────────────────────────────────────────────
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    adata = load_adata()

    ctrl_expr, pert_data, gene_indices = prepare_cell_data(adata, gene_order, device)
    n_genes = ctrl_expr.shape[1]

    # Gene features for the perturbation MLP
    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_features = build_node_features(node_names, adata=adata)
    gene_to_node  = {g: i for i, g in enumerate(node_names)}
    feat_dim      = node_features.shape[1]

    def get_feat(gene):
        idx = gene_to_node.get(gene, 0)
        return torch.FloatTensor(node_features[idx]).to(device)

    # Per-gene evaluation weights (mean across training perturbations)
    gene_weights = torch.FloatTensor(weight_matrix.mean(axis=0)).to(device)

    # ── Model ──────────────────────────────────────────────────────────
    model = CPA(
        n_genes=n_genes,
        latent_dim=latent_dim,
        gene_feat_dim=feat_dim,
        hidden_dims=[512, 256],
        dropout=0.1,
    ).to(device)
    print(f"CPA parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    best_wmae  = float("inf")
    ckpt_path  = os.path.join(config.CHECKPOINTS_DIR, "cpa_best.pt")
    train_perts = [g for g in train_genes if g in pert_data]

    print(f"\nTraining CPA for {n_epochs} epochs …")
    print(f"  Training perturbations with cell data: {len(train_perts)} / {len(train_genes)}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        random.shuffle(train_perts)
        epoch_loss = 0.0

        for pert_gene in train_perts:
            gene_feat   = get_feat(pert_gene)
            pert_cells  = pert_data[pert_gene]           # (n_cells, G)
            n_cells     = pert_cells.shape[0]

            # Sample a batch of control cells and perturbed cells
            ctrl_idx  = torch.randint(0, ctrl_expr.shape[0], (batch_size,), device=device)
            pert_idx  = torch.randint(0, n_cells,            (batch_size,), device=device)
            x_ctrl_b  = ctrl_expr[ctrl_idx]               # (B, G)
            x_pert_b  = pert_cells[pert_idx]              # (B, G)

            optimizer.zero_grad()

            # Forward: predict perturbed expression from control cells + gene features
            x_recon, _, delta = model(x_ctrl_b, gene_feat)

            # Loss 1: reconstruction of perturbed cells
            loss_recon = reconstruction_loss(x_recon, x_pert_b, weights=gene_weights)

            # Loss 2: the perturbation delta should be non-trivial
            # (prevent collapse to δ=0)
            loss_delta = 0.01 * torch.exp(-delta.pow(2).mean())

            loss = loss_recon + loss_delta
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_perts), 1)

        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
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

def predict_cpa(ckpt_path: str | None = None,
                device_str: str = "cuda") -> str:
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
        n_genes=n_genes, latent_dim=64, gene_feat_dim=feat_dim,
        hidden_dims=[512, 256], dropout=0.1,
    ).to(device)

    if ckpt_path is None:
        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "cpa_best.pt")
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

    # Clip to training range
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)
    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_cpa.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nCPA submission saved: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict",    action="store_true")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.predict:
        predict_cpa()
    else:
        ckpt = train_cpa(
            n_epochs=args.epochs,
            latent_dim=args.latent_dim,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        predict_cpa(ckpt_path=ckpt)
