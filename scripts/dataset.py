"""
================================================================================
dataset.py — PyTorch dataset and class-weight utilities for IRD gene data
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
The training data combines three heterogeneous input modalities — a 1905-dimensional
evolutionary conservation profile (NPP) that has been globally z-scored, a
1280-dimensional ESM2 protein language model embedding that is internally
layer-normalised and requires no further scaling, and a 129-dimensional PPI
network embedding derived from STRING interaction profiles. Keeping all three in
a single NPZ file and wrapping them in an IRDDataset ensures that NPP, ESM2, and
PPI features for the same gene are always paired correctly regardless of how
DataLoader shuffles or subsets the data. No additional normalisation is applied
here: the NPP values are already on a comparable scale, ESM2 embeddings are
model-normalised, and PPI embeddings are pre-processed upstream.
Class weights are provided as a separate utility so that they can be computed
exclusively from the training fold, preventing any label-distribution leakage
from the validation fold into the loss function.

Inputs
------
- ../input/training_set.npz: arrays gene_names (434,), npp (434, 1905) float32,
  esm2 (434, 1280) float32, ppi (434, 129) float32, labels (434,) int64 with
  values 0–16.

Outputs
-------
- None (module provides IRDDataset and compute_class_weights for import)

Usage
-----
Imported by train.py and evaluate.py — not run directly.
================================================================================
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IRDDataset(Dataset):
    """PyTorch Dataset wrapping NPP profiles, ESM2 embeddings, PPI embeddings, and IRD module labels for 434 genes."""

    def __init__(self, npz_path: str):
        """Load and dtype-cast all arrays from a pre-built NPZ archive.

        Args:
            npz_path: Absolute or relative path to training_set.npz.

        Creates:
            self.gene_names: (434,) object array of HGNC gene symbols.
            self.npp:        (434, 1905) float32 array of z-scored evolutionary profiles.
            self.esm2:       (434, 1280) float32 array of ESM2 protein embeddings.
            self.ppi:        (434, 129)  float32 array of PPI network embeddings.
            self.labels:     (434,) int64 array of IRD module indices (0–16).
        """
        data = np.load(npz_path, allow_pickle=False)

        self.gene_names = data["gene_names"]
        # float32 is required by PyTorch linear layers; copy=False avoids a
        # redundant allocation when the array is already the correct dtype.
        self.npp = data["npp"].astype(np.float32, copy=False)
        self.esm2 = data["esm2"].astype(np.float32, copy=False)
        self.ppi = data["ppi"].astype(np.float32, copy=False)
        # int64 (torch.long) is required by CrossEntropyLoss as the target dtype.
        self.labels = data["labels"].astype(np.int64, copy=False)

    def __len__(self) -> int:
        """Return the total number of genes in the dataset.

        Returns:
            int: Number of samples (434 for the full training set).
        """
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the NPP profile, ESM2 embedding, PPI embedding, and module label for the gene at index idx.

        Args:
            idx: Integer index into the dataset (0-based).

        Returns:
            npp_tensor:   float32 tensor of shape (1905,).
            esm2_tensor:  float32 tensor of shape (1280,).
            ppi_tensor:   float32 tensor of shape (129,).
            label_tensor: int64 scalar tensor with module index in [0, 16].
        """
        npp_tensor = torch.from_numpy(self.npp[idx])
        esm2_tensor = torch.from_numpy(self.esm2[idx])
        ppi_tensor = torch.from_numpy(self.ppi[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return npp_tensor, esm2_tensor, ppi_tensor, label_tensor


def compute_class_weights(labels: np.ndarray, n_classes: int = 17) -> torch.Tensor:
    """Compute per-class inverse-frequency weights as N / (C × count_c) for use in CrossEntropyLoss.

    Args:
        labels:    1-D int array of class indices drawn from the training fold only.
        n_classes: Number of IRD module classes (default 17).

    Returns:
        float32 tensor of shape (n_classes,) where weight[c] = N / (C × count_c).
        Rare classes receive higher weights, counteracting the class imbalance
        that would otherwise bias the network toward majority modules.
    """
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=n_classes)
    # Guard against zero-count classes: if a class is absent from the training fold
    # (possible when n_classes is large relative to fold size), dividing by zero
    # would produce inf weights and NaN loss.  Clamping to 1 assigns a finite
    # weight to absent classes without distorting the weights of classes that
    # are actually present.
    safe_counts = np.maximum(counts, 1)
    # Inverse-frequency weighting: rare classes penalised more by the loss.
    weights = labels.shape[0] / (n_classes * safe_counts)
    return torch.tensor(weights, dtype=torch.float32)
