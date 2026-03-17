"""
Pairwise deep MLP for perturbation effect prediction.

Architecture
------------
For each (perturbation gene, target gene) pair the input is:

    x_input = [x_pert (D), x_target (D), x_pert ⊙ x_target (D)]  →  3D

Network:
    Linear(3D, 512) → BN → ReLU → Dropout(0.3)
    → Linear(512, 256) → BN → ReLU → Dropout(0.3)
    → Linear(256, 1)

Training expands all (pert, target) pairs:
    80 perts × 5127 genes = 410K training rows (minibatch SGD, batch=2048)

This is a nonlinear generalisation of the pairwise Ridge model.
Unlike the GNN, it has no graph structure — it relies purely on
feature-space similarity between pert and target genes.

Cross-validation is over perturbations (same split as Ridge/LGBM).

Usage
-----
    python src/mlp_model.py
    python src/mlp_model.py --epochs 30 --cv-only
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, load_pert_ids, compute_wmae, load_adata
from graph_builder import build_graph
from node_features import build_node_features
from ridge_baseline import get_gene_features


# ── Model ─────────────────────────────────────────────────────────────────

class PairwiseMLP(nn.Module):
    def __init__(self, in_dim: int,
                 hidden_dims: list[int] = [512, 256],
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


# ── Data helpers ──────────────────────────────────────────────────────────

def build_pairwise_tensors(X_perts: np.ndarray,
                            X_targets: np.ndarray,
                            de_flat: np.ndarray | None = None,
                            w_flat: np.ndarray | None = None):
    """
    Build (X_pair, y, w) tensors from pert × target feature matrices.

    X_perts  : (n_p, D)
    X_targets: (n_t, D)
    de_flat  : (n_p * n_t,) or None
    w_flat   : (n_p * n_t,) or None

    Returns FloatTensors.
    """
    n_p, D = X_perts.shape
    n_t, _ = X_targets.shape

    X_p = np.repeat(X_perts, n_t, axis=0)   # (n_p*n_t, D)
    X_t = np.tile(X_targets, (n_p, 1))       # (n_p*n_t, D)
    inter = X_p * X_t                        # (n_p*n_t, D)

    X = np.concatenate([X_p, X_t, inter], axis=1).astype(np.float32)
    X_t_ = torch.FloatTensor(X)
    if de_flat is not None:
        y_t = torch.FloatTensor(de_flat.astype(np.float32))
        w_t = torch.FloatTensor(w_flat.astype(np.float32))
        return X_t_, y_t, w_t
    return X_t_


# ── Training loop ─────────────────────────────────────────────────────────

def train_mlp(X_perts_tr: np.ndarray,
              X_targets: np.ndarray,
              de_matrix_tr: np.ndarray,
              weight_matrix_tr: np.ndarray,
              device: torch.device,
              epochs: int = 50,
              batch_size: int = 2048,
              lr: float = 3e-4,
              hidden_dims: list[int] = [512, 256],
              dropout: float = 0.3) -> PairwiseMLP:

    D = X_perts_tr.shape[1]
    model = PairwiseMLP(in_dim=3 * D, hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    X_t, y_t, w_t = build_pairwise_tensors(
        X_perts_tr, X_targets,
        de_matrix_tr.flatten(),
        weight_matrix_tr.flatten(),
    )
    ds     = TensorDataset(X_t, y_t, w_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            pred = model(xb)
            loss = (wb * torch.abs(pred - yb)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(loader):.4f}")

    return model


@torch.no_grad()
def predict_mlp(model: PairwiseMLP,
                X_perts: np.ndarray,
                X_targets: np.ndarray,
                device: torch.device,
                batch_size: int = 4096) -> np.ndarray:
    """Returns (n_perts, n_targets) float32 predictions."""
    model.eval()
    X_t = build_pairwise_tensors(X_perts, X_targets)
    ds  = DataLoader(TensorDataset(X_t), batch_size=batch_size)
    preds = []
    for (xb,) in ds:
        preds.append(model(xb.to(device)).cpu().numpy())
    flat = np.concatenate(preds)
    return flat.reshape(X_perts.shape[0], X_targets.shape[0])


# ── Main ──────────────────────────────────────────────────────────────────

def run_mlp(epochs: int = 50,
            batch_size: int = 2048,
            lr: float = 3e-4,
            hidden_dims: list[int] = [512, 256],
            dropout: float = 0.3,
            cv_only: bool = False) -> str:
    """
    Train pairwise MLP → submission CSV.
    Returns submission path.
    """
    config.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graph + node features
    adata      = load_adata()
    node_names, _, _ = build_graph(score_threshold=config.STRING_SCORE_GNN, adata=adata)
    node_feats = build_node_features(node_names, adata=adata)

    # Load DE + weights
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()
    n_t = len(gene_order)

    X_perts_train = get_gene_features(train_genes, node_names, node_feats)  # (80, D)
    X_targets     = get_gene_features(gene_order,  node_names, node_feats)  # (n_t, D)

    # Scale all gene features together
    all_feats  = np.vstack([X_perts_train, X_targets])
    scaler     = StandardScaler().fit(all_feats)
    X_p_sc     = scaler.transform(X_perts_train)
    X_t_sc     = scaler.transform(X_targets)

    print(f"X_perts: {X_p_sc.shape}  X_targets: {X_t_sc.shape}")
    print(f"Training pairs: {X_p_sc.shape[0] * n_t:,}")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    oof_preds = np.zeros_like(de_matrix)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_p_sc)):
        print(f"\n[Fold {fold+1}] train={len(tr_idx)}, val={len(va_idx)}")
        model = train_mlp(
            X_p_sc[tr_idx], X_t_sc,
            de_matrix[tr_idx], weight_matrix[tr_idx],
            device=device, epochs=epochs, batch_size=batch_size,
            lr=lr, hidden_dims=hidden_dims, dropout=dropout,
        )
        oof_preds[va_idx] = predict_mlp(model, X_p_sc[va_idx], X_t_sc, device)
        fold_wmae = compute_wmae(oof_preds[va_idx], de_matrix[va_idx], weight_matrix[va_idx])
        print(f"  Fold {fold+1} WMAE: {fold_wmae:.4f}")

    oof_wmae = compute_wmae(oof_preds, de_matrix, weight_matrix)
    print(f"\nOOF WMAE: {oof_wmae:.4f}  (baseline: 0.1268)")

    oof_df = pd.DataFrame(oof_preds, columns=gene_order)
    oof_df.insert(0, "pert_id", train_genes)
    oof_path = os.path.join(config.SUBMISSIONS_DIR, "oof_mlp.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    if cv_only:
        return oof_path

    # Final model on all 80 training perts
    print("\nFinal training on all 80 perturbations …")
    final_model = train_mlp(
        X_p_sc, X_t_sc, de_matrix, weight_matrix,
        device=device, epochs=epochs, batch_size=batch_size,
        lr=lr, hidden_dims=hidden_dims, dropout=dropout,
    )
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "mlp_best.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"Checkpoint: {ckpt_path}")

    # Test prediction
    test_df    = load_pert_ids()
    test_genes = test_df["pert"].tolist()
    test_ids   = test_df["pert_id"].tolist()

    X_perts_test = get_gene_features(test_genes, node_names, node_feats)
    X_perts_test = scaler.transform(X_perts_test)

    preds = predict_mlp(final_model, X_perts_test, X_t_sc, device)

    # Clip to training range per gene
    gene_min = de_matrix.min(axis=0)
    gene_max = de_matrix.max(axis=0)
    preds = np.clip(preds, gene_min, gene_max).astype(np.float32)

    sub = pd.DataFrame(preds, columns=gene_order)
    sub.insert(0, "pert_id", test_ids)
    out_path = os.path.join(config.SUBMISSIONS_DIR, "submission_mlp.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nMLP submission saved: {out_path}")
    print(f"mean|pred|: {np.abs(preds).mean():.4f}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--cv-only",    action="store_true")
    args = parser.parse_args()

    run_mlp(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        cv_only=args.cv_only,
    )
