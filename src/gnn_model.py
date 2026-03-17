"""
GEARS-style Graph Attention Network for gene perturbation prediction.

Architecture (two-stage):
  Stage 1: Context Encoder  — GNN that learns gene representations
            from the static gene interaction graph (runs once per epoch)
  Stage 2: Perturbation Propagator — propagates the perturbation signal
            through the graph using the base embeddings as context

Key design: the perturbation signal is derived from the perturbed gene's own
embedding h_base[pert_idx], enabling zero-shot generalisation to unseen genes.
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
    print("WARNING: torch-geometric not installed. GNN model will not be available.")


class InputProjection(nn.Module):
    """Project raw node features to the GNN hidden dimension."""
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
    """
    Single GAT block: multi-head attention + residual + LayerNorm.
    Input and output both have dimension `hidden_dim`.
    """
    def __init__(self, hidden_dim: int, heads: int, dropout: float,
                 in_dim: int = None):
        super().__init__()
        in_dim = in_dim or hidden_dim
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=1,          # scalar edge weight
            concat=True,         # concatenate heads → hidden_dim
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Residual projection if in_dim != hidden_dim
        if in_dim != hidden_dim:
            self.residual_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index, edge_attr):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        out = self.conv(x, edge_index, edge_attr.unsqueeze(-1))
        out = self.dropout(out)
        out = self.norm(out + residual)
        return self.activation(out)


class GEARSModel(nn.Module):
    """
    GEARS-style GNN for perturbation effect prediction.

    Parameters
    ----------
    node_feat_dim : int  – raw node feature dimension (default 119 = 5+50+64)
    hidden_dim    : int  – GNN hidden dimension
    num_layers    : int  – number of GAT layers in each stage
    output_dim    : int  – number of output genes (5127)
    heads         : int  – number of attention heads per GAT layer
    dropout       : float
    """

    def __init__(self,
                 node_feat_dim: int = 119,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 output_dim: int = 5127,
                 heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        assert HAS_PYG, "torch-geometric is required for GEARSModel"

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = InputProjection(node_feat_dim, hidden_dim, dropout=dropout)

        # ── Stage 1: Context encoder (perturbation-independent) ───────────
        self.encoder = nn.ModuleList([
            GATBlock(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Stage 2: Perturbation signal transform ────────────────────────
        # Maps h_base[pert_idx] → perturbation signal vector
        self.pert_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # ── Stage 2: Perturbation propagation layers ──────────────────────
        # Input: concat(h_base, pert_signal) → hidden_dim * 2
        self.pert_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.pert_propagator = nn.ModuleList([
            GATBlock(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        ])

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   # 1 DE value per gene
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Stage 1 ───────────────────────────────────────────────────────────

    def encode_graph(self, x: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute context-aware gene embeddings (perturbation-independent).
        This can be called once per epoch and the result used for all
        perturbations within that epoch.

        Parameters
        ----------
        x          : (n_nodes, node_feat_dim)
        edge_index : (2, n_edges)
        edge_attr  : (n_edges,)

        Returns
        -------
        h_base : (n_nodes, hidden_dim)
        """
        h = self.input_proj(x)
        for layer in self.encoder:
            h = layer(h, edge_index, edge_attr)
        return h

    # ── Stage 2 ───────────────────────────────────────────────────────────

    def forward_perturbation(self,
                             h_base: torch.Tensor,
                             pert_gene_idx: int,
                             edge_index: torch.Tensor,
                             edge_attr: torch.Tensor,
                             output_gene_indices: list[int]) -> torch.Tensor:
        """
        Propagate perturbation signal and predict DE for output genes.

        Parameters
        ----------
        h_base            : (n_nodes, hidden_dim) – from encode_graph()
        pert_gene_idx     : int – node index of the perturbed gene
        edge_index        : (2, n_edges)
        edge_attr         : (n_edges,)
        output_gene_indices: list of n_output_genes node indices

        Returns
        -------
        de_pred : (n_output_genes,) – predicted differential expression
        """
        n_nodes = h_base.size(0)

        # Derive perturbation signal from the perturbed gene's embedding
        pert_emb    = h_base[pert_gene_idx]                    # (hidden_dim,)
        pert_signal = self.pert_transform(pert_emb)             # (hidden_dim,)

        # Broadcast: each gene node gets the same perturbation signal appended
        pert_broadcast = pert_signal.unsqueeze(0).expand(n_nodes, -1)  # (N, hidden_dim)
        h_combined = torch.cat([h_base, pert_broadcast], dim=-1)       # (N, 2*hidden_dim)

        # Project combined input back to hidden_dim
        h = self.pert_proj(h_combined)                                   # (N, hidden_dim)

        # Propagate through perturbation layers with residual from h_base
        for layer in self.pert_propagator:
            h = layer(h, edge_index, edge_attr)
            h = h + h_base * 0.1   # soft residual from base to prevent drift

        # Decode: run only on output gene nodes
        out_idx = torch.tensor(output_gene_indices, dtype=torch.long, device=h.device)
        output_embeddings = h[out_idx]                         # (n_output, hidden_dim)
        de_pred = self.decoder(output_embeddings).squeeze(-1)  # (n_output,)

        return de_pred

    def forward(self, x, edge_index, edge_attr, pert_gene_idx, output_gene_indices):
        """
        Full forward pass for a single perturbation.
        Calls encode_graph then forward_perturbation.
        Use this for inference; use encode_graph + forward_perturbation separately
        for training (encode once per epoch, then iterate perturbations).
        """
        h_base = self.encode_graph(x, edge_index, edge_attr)
        return self.forward_perturbation(h_base, pert_gene_idx,
                                         edge_index, edge_attr, output_gene_indices)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss function ─────────────────────────────────────────────────────────

def weighted_mae_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted Mean Absolute Error — matches competition evaluation metric.

    pred    : (n_genes,) or (batch, n_genes)
    target  : same shape as pred
    weights : same shape as pred
    """
    return (weights * torch.abs(pred - target)).mean()


def weighted_huber_loss(pred: torch.Tensor,
                         target: torch.Tensor,
                         weights: torch.Tensor,
                         delta: float = 0.05) -> torch.Tensor:
    """
    Weighted Huber loss — smoother than MAE near zero, useful for warm-up.
    Use this for the first few epochs, then switch to weighted_mae_loss.
    """
    diff = torch.abs(pred - target)
    huber = torch.where(diff < delta,
                        0.5 * diff ** 2 / delta,
                        diff - 0.5 * delta)
    return (weights * huber).mean()


# ── DropEdge ──────────────────────────────────────────────────────────────

def drop_edges(edge_index: torch.Tensor,
               edge_attr: torch.Tensor,
               drop_rate: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop edges during training (data augmentation)."""
    if drop_rate == 0:
        return edge_index, edge_attr
    n_edges = edge_index.size(1)
    keep_mask = torch.rand(n_edges, device=edge_index.device) > drop_rate
    return edge_index[:, keep_mask], edge_attr[keep_mask]


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import config

    if not HAS_PYG:
        print("torch-geometric not installed. Please run: pip install torch-geometric")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dummy data
    n_nodes    = 5400
    n_edges    = 50000
    n_output   = 5127
    feat_dim   = config.NODE_FEAT_DIM

    x          = torch.randn(n_nodes, feat_dim).to(device)
    edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(device)
    edge_attr  = torch.rand(n_edges).to(device)
    output_idx = list(range(n_output))
    pert_idx   = 42

    model = GEARSModel(
        node_feat_dim=feat_dim,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        output_dim=n_output,
        heads=config.GNN_HEADS,
        dropout=config.GNN_DROPOUT,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Forward pass
    h_base = model.encode_graph(x, edge_index, edge_attr)
    print(f"h_base shape: {h_base.shape}")  # (n_nodes, hidden_dim)

    de_pred = model.forward_perturbation(h_base, pert_idx, edge_index, edge_attr, output_idx)
    print(f"de_pred shape: {de_pred.shape}")  # (n_output,)

    # Loss
    target  = torch.randn(n_output).to(device)
    weights = torch.rand(n_output).to(device)
    loss = weighted_mae_loss(de_pred, target, weights)
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward pass: OK")
