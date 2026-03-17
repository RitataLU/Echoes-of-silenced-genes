"""
PyTorch Dataset for perturbation data.

The gene interaction graph is kept as a static GPU object (not part of the
dataset items), because all perturbations share the same graph. Only the
per-perturbation items (pert_idx, de, weights) are returned by __getitem__.
"""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import List, Optional
import numpy as np


class PerturbationDataset(Dataset):
    """
    Dataset for gene perturbation prediction.

    Parameters
    ----------
    pert_genes          : list of gene names to perturb (80 train or 120 test)
    pert_ids            : list of pert_id strings matching pert_genes order
    gene_to_node_idx    : dict mapping gene name → node index in graph
    output_gene_indices : list of node indices for the 5127 output genes
    de_matrix           : (n_perts, n_genes) float32 – ground truth DE (None for test)
    weight_matrix       : (n_perts, n_genes) float32 – per-gene weights (None for test)
    """

    def __init__(self,
                 pert_genes: List[str],
                 pert_ids: List[str],
                 gene_to_node_idx: dict,
                 output_gene_indices: List[int],
                 de_matrix: Optional[np.ndarray] = None,
                 weight_matrix: Optional[np.ndarray] = None):

        self.pert_genes          = pert_genes
        self.pert_ids            = pert_ids
        self.gene_to_node_idx    = gene_to_node_idx
        self.output_gene_indices = output_gene_indices
        self.de_matrix           = de_matrix
        self.weight_matrix       = weight_matrix
        self.is_train            = de_matrix is not None

        # Pre-compute node indices for all perturbation genes
        self.pert_node_indices = []
        missing = []
        for gene in pert_genes:
            idx = gene_to_node_idx.get(gene)
            if idx is None:
                missing.append(gene)
                idx = 0   # fallback — should not happen if graph is built correctly
            self.pert_node_indices.append(idx)

        if missing:
            print(f"WARNING: {len(missing)} perturbation genes not found in graph: {missing}")

    def __len__(self) -> int:
        return len(self.pert_genes)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "pert_gene"    : self.pert_genes[idx],
            "pert_id"      : self.pert_ids[idx],
            "pert_node_idx": self.pert_node_indices[idx],
        }
        if self.is_train:
            item["de"]      = torch.FloatTensor(self.de_matrix[idx])
            item["weights"] = torch.FloatTensor(self.weight_matrix[idx])
        return item

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Custom collate: keeps pert_node_idx as a plain list of ints
        (not a batched tensor) since each perturbation needs its own
        forward_perturbation call.
        """
        keys = batch[0].keys()
        result = {}
        for k in keys:
            if k in ("pert_gene", "pert_id"):
                result[k] = [item[k] for item in batch]
            elif k == "pert_node_idx":
                result[k] = [item[k] for item in batch]
            else:
                result[k] = torch.stack([item[k] for item in batch])
        return result
