"""
GEARSModel v2 — Gene-Program head + MCDropout TTA.

Changes from v1
---------------
1. Program head: decoder predicts K program scores instead of 5127 DE values.
   Final DE = predicted_scores @ H  (H = gene program matrix, (K, G))
   This dramatically reduces the output dimensionality and constrains
   predictions to biologically meaningful co-expression patterns.

2. MCDropout TTA: keep dropout ON at inference time and average N forward
   passes → free uncertainty-based ensemble (Gal & Ghahramani, 2016).

3. Deeper decoder MLP: Linear → ReLU → Dropout → Linear (→ K scores).

4. Learnable perturbation residual: a small MLP that adds a direct skip
   connection from the perturbed gene's raw features to the output,
   preventing the GNN from forgetting which gene was knocked down.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch-geometric not installed.")


# ── Reuse building blocks from v1 ─────────────────────────────────────────

class InputProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class GATBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, in_dim: int = None):
        super().__init__()
        in_dim = in_dim or hidden_dim
        assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads

        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            concat=True,
            add_self_loops=True,
        )
        self.norm      = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.residual_proj = nn.Linear(in_dim, hidden_dim, bias=False) \
            if in_dim != hidden_dim else None

    def forward(self, x, edge_index, edge_attr):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = self.conv(x, edge_index, edge_attr.unsqueeze(-1))
        out = self.dropout(out)
        return self.activation(self.norm(out + residual))


# ── v2 Model ──────────────────────────────────────────────────────────────

class GEARSModelV2(nn.Module):
    """
    GEARS-style GNN with gene-program output head and MCDropout TTA.

    Parameters
    ----------
    node_feat_dim : int   raw node feature dim (e.g. 119 or 1163 with ESM2)
    hidden_dim    : int   GNN hidden dim (256 on GPU)
    num_layers    : int   GAT layers per stage (3 on GPU)
    K             : int   number of gene programs (64)
    heads         : int   GAT attention heads (4 on GPU)
    dropout       : float
    """

    def __init__(self,
                 node_feat_dim: int = 119,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 K: int = 64,
                 heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        assert HAS_PYG, "torch-geometric required"

        self.hidden_dim = hidden_dim
        self.K = K

        # ── Input projection ──────────────────────────────────────────
        self.input_proj = InputProjection(node_feat_dim, hidden_dim, dropout)

        # ── Stage 1: Context Encoder ──────────────────────────────────
        self.encoder = nn.ModuleList([
            GATBlock(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Stage 2: Perturbation signal transform ────────────────────
        self.pert_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Perturbation propagation layers
        self.pert_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.pert_propagator = nn.ModuleList([
            GATBlock(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Global pooling for perturbed gene context ─────────────────
        # After propagation, pool the perturbed gene's final embedding
        # and add it as an extra signal for the decoder.
        self.pert_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Decoder → K program scores ────────────────────────────────
        # Input: mean of all output-gene embeddings (global context)
        #        + perturbed gene's final embedding
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, K),
        )

        # ── Learnable direct residual from pert gene raw features ─────
        # Gives the model a shortcut to use the perturbed gene's node
        # features directly, rather than relying purely on graph propagation.
        self.raw_residual = nn.Linear(node_feat_dim, K, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Stage 1 ───────────────────────────────────────────────────────

    def encode_graph(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for layer in self.encoder:
            h = layer(h, edge_index, edge_attr)
        return h   # (N, hidden_dim)

    # ── Stage 2 ───────────────────────────────────────────────────────

    def forward_perturbation(self,
                             x_raw: torch.Tensor,
                             h_base: torch.Tensor,
                             pert_gene_idx: int,
                             edge_index: torch.Tensor,
                             edge_attr: torch.Tensor,
                             output_gene_indices: list[int]) -> torch.Tensor:
        """
        Returns predicted program scores: (K,)
        """
        n_nodes = h_base.size(0)

        # Perturbation signal from perturbed gene's base embedding
        pert_emb    = h_base[pert_gene_idx]                            # (H,)
        pert_signal = self.pert_transform(pert_emb)                    # (H,)

        pert_broadcast = pert_signal.unsqueeze(0).expand(n_nodes, -1)  # (N, H)
        h_combined = torch.cat([h_base, pert_broadcast], dim=-1)       # (N, 2H)
        h = self.pert_proj(h_combined)                                  # (N, H)

        for layer in self.pert_propagator:
            h = layer(h, edge_index, edge_attr)
            h = h + h_base * 0.1   # soft residual

        # ── Decode via program scores ──────────────────────────────────

        # 1. Pool embeddings of output genes (global representation)
        out_idx = torch.tensor(output_gene_indices, dtype=torch.long, device=h.device)
        out_embeddings = h[out_idx]                          # (G, H)
        global_ctx = out_embeddings.mean(dim=0)              # (H,) — mean pool

        # 2. Gated perturbed-gene embedding
        pert_final  = h[pert_gene_idx]                       # (H,)
        gate        = self.pert_gate(pert_final)             # (H,) ∈ (0,1)
        pert_gated  = pert_final * gate                      # (H,)

        # 3. Decode to K scores
        dec_input   = torch.cat([global_ctx, pert_gated], dim=-1)  # (2H,)
        w_pred      = self.decoder(dec_input)                       # (K,)

        # 4. Raw-feature residual (direct shortcut for the perturbed gene)
        w_raw       = self.raw_residual(x_raw[pert_gene_idx])       # (K,)

        return w_pred + 0.1 * w_raw                                 # (K,)

    def forward(self, x, edge_index, edge_attr, pert_gene_idx, output_gene_indices):
        h_base = self.encode_graph(x, edge_index, edge_attr)
        return self.forward_perturbation(
            x, h_base, pert_gene_idx, edge_index, edge_attr, output_gene_indices
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MCDropout TTA ─────────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_predict(model: GEARSModelV2,
                       x: torch.Tensor,
                       edge_index: torch.Tensor,
                       edge_attr: torch.Tensor,
                       pert_gene_idx: int,
                       output_gene_indices: list[int],
                       H: torch.Tensor,
                       n_samples: int = 30) -> torch.Tensor:
    """
    MCDropout test-time augmentation: keep model in TRAIN mode (so dropout
    is active) and average N forward passes.

    Returns
    -------
    de_pred : (n_output_genes,) — averaged DE prediction
    """
    model.train()   # dropout ON

    preds = []
    for _ in range(n_samples):
        # Re-encode graph each time (different dropout masks)
        h_base = model.encode_graph(x, edge_index, edge_attr)
        w = model.forward_perturbation(
            x, h_base, pert_gene_idx, edge_index, edge_attr, output_gene_indices
        )
        de = w @ H                    # (G,)
        preds.append(de)

    return torch.stack(preds).mean(0)   # (G,)


# ── Loss functions ─────────────────────────────────────────────────────────

def weighted_mae_loss_v2(w_pred: torch.Tensor,
                         w_true: torch.Tensor,
                         H: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """
    Decode program scores to DE and compute weighted MAE.
    Loss is in DE space to exactly match the competition metric.
    """
    de_pred = w_pred @ H    # (G,)
    de_true = w_true @ H    # (G,)
    return (weights * torch.abs(de_pred - de_true)).mean()


def weighted_huber_loss_v2(w_pred, w_true, H, weights, delta=0.05):
    de_pred = w_pred @ H
    de_true = w_true @ H
    diff    = torch.abs(de_pred - de_true)
    huber   = torch.where(diff < delta, 0.5 * diff**2 / delta, diff - 0.5 * delta)
    return (weights * huber).mean()


def drop_edges(edge_index, edge_attr, drop_rate=0.10):
    if drop_rate == 0:
        return edge_index, edge_attr
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep], edge_attr[keep]


# ── Sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import config

    if not HAS_PYG:
        print("torch-geometric not installed.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    N, E, G, K = 5400, 50000, 5127, config.GENE_PROGRAM_K
    x          = torch.randn(N, config.NODE_FEAT_DIM).to(device)
    ei         = torch.randint(0, N, (2, E)).to(device)
    ea         = torch.rand(E).to(device)
    H_mat      = torch.randn(K, G).to(device)
    out_idx    = list(range(G))
    pert_idx   = 42

    model = GEARSModelV2(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        K=K,
        heads=config.GNN_HEADS,
        dropout=config.GNN_DROPOUT,
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")

    h_base = model.encode_graph(x, ei, ea)
    w_pred = model.forward_perturbation(x, h_base, pert_idx, ei, ea, out_idx)
    print(f"w_pred shape: {w_pred.shape}")   # (K,)
    de_pred = w_pred @ H_mat
    print(f"de_pred shape: {de_pred.shape}") # (G,)

    w_true  = torch.randn(K).to(device)
    weights = torch.rand(G).to(device)
    loss = weighted_mae_loss_v2(w_pred, w_true, H_mat, weights)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print("Backward: OK")

    # MCDropout
    de_tta = mc_dropout_predict(model, x, ei, ea, pert_idx, out_idx, H_mat, n_samples=5)
    print(f"TTA de shape: {de_tta.shape}")
