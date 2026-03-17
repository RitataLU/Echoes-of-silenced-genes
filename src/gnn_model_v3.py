"""
GEARSModel v3 — GraphTransformer + directed TF edges + global context.

Improvements over v2
--------------------
1. TransformerConv instead of GATConv
   - Dot-product attention (Q·K/√d) scales better than additive GAT attention
   - Learns edge-feature modulated attention (edge_dim parameter)
   - Learns a per-layer β scalar for mixing skip-connection (PyG beta=True)

2. Edge-type encoding
   - Separate learnable embedding for edge types (PPI/coexp=0, TF-act=1, TF-rep=2)
   - Combined with edge weight → projected to edge_feat_dim per layer
   - Lets the model treat TF→target signal differently from PPI

3. Global context layer (VirtualNode-style, O(N))
   - Reads a global graph summary via mean-pool + gating
   - Writes back to all nodes (soft residual)
   - Captures long-range correlations without O(N²) self-attention
   - Enabled via use_global_ctx=True (default False for CPU)

Everything else (gene-program output head, MCDropout TTA, gated decoder,
raw-feature residual) is unchanged from v2.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch-geometric not installed.")


# ── Edge encoder ──────────────────────────────────────────────────────────

class EdgeEncoder(nn.Module):
    """
    Project raw edge features (E, 2) → (E, edge_feat_dim).

    Input channels:
        col 0: edge weight  (0–1)
        col 1: edge type id (0, 1, 2)
    """
    N_TYPES = 3   # PPI/coexp, TF_activating, TF_repressing

    def __init__(self, edge_feat_dim: int):
        super().__init__()
        # Learnable type embeddings
        self.type_emb = nn.Embedding(self.N_TYPES, edge_feat_dim // 2)
        # Project [weight_proj + type_emb] → edge_feat_dim
        self.weight_proj = nn.Linear(1, edge_feat_dim // 2)
        self.out_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_feat_dim),
            nn.ReLU(),
        )

    def forward(self, edge_attr_2d: torch.Tensor) -> torch.Tensor:
        """
        edge_attr_2d : (E, 2) — [weight, type_float]
        returns      : (E, edge_feat_dim)
        """
        weight = edge_attr_2d[:, 0:1]                               # (E, 1)
        etype  = edge_attr_2d[:, 1].long().clamp(0, self.N_TYPES-1) # (E,) int
        w_enc  = self.weight_proj(weight)       # (E, edge_feat_dim//2)
        t_enc  = self.type_emb(etype)           # (E, edge_feat_dim//2)
        return self.out_proj(torch.cat([w_enc, t_enc], dim=-1))     # (E, edge_feat_dim)


# ── Global context (virtual-node style) ───────────────────────────────────

class GlobalContextLayer(nn.Module):
    """
    O(N) long-range context.
    1. Reads a graph summary: g = mean_pool(h) → Linear → g'
    2. Broadcasts g' to every node with a learned gate.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.read   = nn.Linear(hidden_dim, hidden_dim)
        self.write  = nn.Linear(hidden_dim, hidden_dim)
        self.gate   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm   = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g  = F.relu(self.read(h.mean(0)))                           # (H,)
        g_bc = self.write(g).unsqueeze(0).expand_as(h)              # (N, H)
        gate = torch.sigmoid(self.gate(torch.cat([h, g_bc], dim=-1))) # (N, H)
        return self.norm(h + self.drop(gate * g_bc))


# ── Transformer block ──────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    TransformerConv (local message-passing) + optional GlobalContextLayer.

    Output dim is always `hidden_dim` regardless of `in_dim`.
    """
    def __init__(self,
                 hidden_dim: int,
                 heads: int,
                 dropout: float,
                 edge_feat_dim: int,
                 in_dim: int = None,
                 use_global_ctx: bool = False):
        super().__init__()
        in_dim = in_dim or hidden_dim
        assert hidden_dim % heads == 0
        out_per_head = hidden_dim // heads

        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_per_head,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=True,
            beta=True,          # learnable skip-connection weight
        )
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ReLU()

        self.residual_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim else None
        )

        self.global_ctx = GlobalContextLayer(hidden_dim, dropout) if use_global_ctx else None

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_feats: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = self.conv(x, edge_index, edge_feats)
        out = self.act(self.norm(self.dropout(out) + residual))
        if self.global_ctx is not None:
            out = self.global_ctx(out)
        return out


# ── Input projection (same as v2) ─────────────────────────────────────────

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


# ── v3 Model ───────────────────────────────────────────────────────────────

class GEARSModelV3(nn.Module):
    """
    GEARS-style GNN v3: TransformerConv + directed TF edges + global context.

    Differences from v2
    -------------------
    - GATConv replaced by TransformerConv (dot-product attention)
    - EdgeEncoder projects (E, 2) edge features to edge_feat_dim
    - Optional GlobalContextLayer after each TransformerBlock
    - encode_edges() method to pre-compute edge features per batch

    Interface
    ---------
    edge_attr passed here must be (E, 2) float32 — [weight, type].
    Call encode_edges() once per batch, then pass the result to
    encode_graph() and forward_perturbation().
    """

    def __init__(self,
                 node_feat_dim: int = 119,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 K: int = 64,
                 heads: int = 4,
                 dropout: float = 0.2,
                 edge_feat_dim: int = 32,
                 use_global_ctx: bool = False):
        super().__init__()
        assert HAS_PYG, "torch-geometric required"

        self.hidden_dim   = hidden_dim
        self.K            = K
        self.edge_feat_dim = edge_feat_dim

        # ── Edge encoder ──────────────────────────────────────────────
        self.edge_encoder = EdgeEncoder(edge_feat_dim)

        # ── Input projection ──────────────────────────────────────────
        self.input_proj = InputProjection(node_feat_dim, hidden_dim, dropout)

        # ── Stage 1: Context Encoder ──────────────────────────────────
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim
            self.encoder.append(
                TransformerBlock(hidden_dim, heads, dropout, edge_feat_dim,
                                 in_dim=in_d, use_global_ctx=use_global_ctx)
            )

        # ── Stage 2: Perturbation propagation ─────────────────────────
        self.pert_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pert_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.pert_propagator = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dropout, edge_feat_dim,
                             use_global_ctx=use_global_ctx)
            for _ in range(num_layers)
        ])

        # Gated perturbed-gene embedding
        self.pert_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Decoder → K program scores ────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, K),
        )

        # Raw feature residual shortcut
        self.raw_residual = nn.Linear(node_feat_dim, K, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Edge encoding (call once per batch) ───────────────────────────

    def encode_edges(self, edge_attr_2d: torch.Tensor) -> torch.Tensor:
        """
        Project raw (E, 2) edge features to (E, edge_feat_dim).
        Call this once per batch, then pass result to encode_graph /
        forward_perturbation.
        """
        return self.edge_encoder(edge_attr_2d)

    # ── Stage 1 ───────────────────────────────────────────────────────

    def encode_graph(self,
                     x: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_feats: torch.Tensor) -> torch.Tensor:
        """
        edge_feats: pre-computed via encode_edges(), shape (E, edge_feat_dim)
        """
        h = self.input_proj(x)
        for layer in self.encoder:
            h = layer(h, edge_index, edge_feats)
        return h

    # ── Stage 2 ───────────────────────────────────────────────────────

    def forward_perturbation(self,
                             x_raw: torch.Tensor,
                             h_base: torch.Tensor,
                             pert_gene_idx: int,
                             edge_index: torch.Tensor,
                             edge_feats: torch.Tensor,
                             output_gene_indices: list[int]) -> torch.Tensor:
        """Returns predicted program scores: (K,)"""
        n_nodes = h_base.size(0)

        pert_emb    = h_base[pert_gene_idx]
        pert_signal = self.pert_transform(pert_emb)
        pert_bc     = pert_signal.unsqueeze(0).expand(n_nodes, -1)
        h_combined  = torch.cat([h_base, pert_bc], dim=-1)
        h           = self.pert_proj(h_combined)

        for layer in self.pert_propagator:
            h = layer(h, edge_index, edge_feats)
            h = h + h_base * 0.1   # soft skip

        # Decode
        out_idx     = torch.tensor(output_gene_indices, dtype=torch.long, device=h.device)
        global_ctx  = h[out_idx].mean(0)
        pert_final  = h[pert_gene_idx]
        gate        = self.pert_gate(pert_final)
        pert_gated  = pert_final * gate

        dec_input   = torch.cat([global_ctx, pert_gated], dim=-1)
        w_pred      = self.decoder(dec_input)
        w_raw       = self.raw_residual(x_raw[pert_gene_idx])

        return w_pred + 0.1 * w_raw   # (K,)

    def forward(self, x, edge_index, edge_attr_2d, pert_gene_idx, output_gene_indices):
        edge_feats = self.encode_edges(edge_attr_2d)
        h_base     = self.encode_graph(x, edge_index, edge_feats)
        return self.forward_perturbation(
            x, h_base, pert_gene_idx, edge_index, edge_feats, output_gene_indices
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MCDropout TTA ─────────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_predict_v3(model: GEARSModelV3,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_attr_2d: torch.Tensor,
                           pert_gene_idx: int,
                           output_gene_indices: list[int],
                           H: torch.Tensor,
                           n_samples: int = 10) -> torch.Tensor:
    """MCDropout TTA — keeps dropout ON, averages N forward passes."""
    model.train()
    preds = []
    edge_feats = model.encode_edges(edge_attr_2d)   # encode once (no dropout on edges)
    for _ in range(n_samples):
        h_base = model.encode_graph(x, edge_index, edge_feats)
        w = model.forward_perturbation(
            x, h_base, pert_gene_idx, edge_index, edge_feats, output_gene_indices
        )
        preds.append(w @ H)
    return torch.stack(preds).mean(0)


# ── Loss functions (identical to v2, re-exported for convenience) ──────────

def weighted_mae_loss_v3(w_pred, w_true, H, weights):
    de_pred = w_pred @ H
    de_true = w_true @ H
    return (weights * torch.abs(de_pred - de_true)).mean()


def weighted_huber_loss_v3(w_pred, w_true, H, weights, delta=0.05):
    de_pred = w_pred @ H
    de_true = w_true @ H
    diff    = torch.abs(de_pred - de_true)
    huber   = torch.where(diff < delta, 0.5 * diff**2 / delta, diff - 0.5 * delta)
    return (weights * huber).mean()


def drop_edges(edge_index, edge_attr, drop_rate=0.10):
    """Works for both 1D (E,) and 2D (E, F) edge_attr."""
    if drop_rate == 0:
        return edge_index, edge_attr
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep], edge_attr[keep]


# ── Sanity check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import config_local as config

    if not HAS_PYG:
        print("torch-geometric not installed. Exiting.")
        sys.exit(1)

    device = torch.device("cpu")
    N, E, G, K = 200, 2000, 100, config.GENE_PROGRAM_K

    x      = torch.randn(N, config.NODE_FEAT_DIM)
    ei     = torch.randint(0, N, (2, E))
    ea_2d  = torch.stack([torch.rand(E), torch.randint(0, 3, (E,)).float()], dim=1)
    H_mat  = torch.randn(K, G)
    out_idx = list(range(G))

    model = GEARSModelV3(
        node_feat_dim=config.NODE_FEAT_DIM,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        K=K,
        heads=config.GNN_HEADS,
        dropout=config.GNN_DROPOUT,
        edge_feat_dim=32,
        use_global_ctx=False,
    )
    print(f"Parameters: {model.count_parameters():,}")

    edge_feats = model.encode_edges(ea_2d)
    h_base     = model.encode_graph(x, ei, edge_feats)
    w_pred     = model.forward_perturbation(x, h_base, 42, ei, edge_feats, out_idx)
    print(f"w_pred: {w_pred.shape}")

    de  = w_pred @ H_mat
    print(f"de:     {de.shape}")

    loss = weighted_mae_loss_v3(w_pred, torch.randn(K), H_mat, torch.rand(G))
    loss.backward()
    print(f"Loss: {loss.item():.4f}  Backward: OK")
