"""
================================================================================
model.py — Three-tower fusion network for IRD gene module classification
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
Classifying IRD genes into 17 phenotypic modules benefits from combining
complementary biological signals: the NPP evolutionary conservation profile
captures cross-species constraint patterns, the ESM2 protein language model
embedding encodes sequence-level biochemical context, and the PPI network
embedding encodes protein–protein interaction neighbourhood structure. A
three-tower architecture processes each modality independently before fusion,
allowing each tower to learn modality-specific representations without
interference. The shared fusion head then learns how to integrate all three
signals for the classification task. Seven variants (fusion_3tower, npp_only,
esm2_only, ppi_only, npp_esm2, npp_ppi, esm2_ppi) are supported by the same
class, enabling ablation experiments that quantify each modality's marginal
contribution. ESM2 and PPI embeddings are treated as fixed pre-computed inputs;
neither model is instantiated here because their inference was run offline and
the resulting embeddings are stored in the training NPZ file.

Inputs
------
- npp:  float32 tensor of shape (batch, 1905) — z-scored evolutionary profiles
- esm2: float32 tensor of shape (batch, 1280) — pre-computed ESM2 embeddings
- ppi:  float32 tensor of shape (batch, 129)  — pre-computed PPI network embeddings

Outputs
-------
- forward():       float32 tensor of shape (batch, 17) — class logits
- get_embedding(): float32 tensor of shape (batch, 64) — fusion representation

Usage
-----
Imported by train.py and evaluate.py — not run directly.
================================================================================
"""

from __future__ import annotations

import torch
from torch import nn


class ThreeTowerClassifier(nn.Module):
    """Three-tower fusion classifier combining NPP evolutionary profiles (1905-dim), ESM2 embeddings (1280-dim), and PPI network embeddings (129-dim) for 17-class IRD module prediction."""

    def __init__(self, use_npp: bool = True, use_esm2: bool = True, use_ppi: bool = True, dropout: float = 0.3):
        """Build modality towers and fusion head according to the requested variant.

        Args:
            use_npp:  If True, include the NPP tower (Linear 1905→64 + ReLU + Dropout).
            use_esm2: If True, include the ESM2 tower (Linear 1280→128 + ReLU + Dropout).
            use_ppi:  If True, include the PPI tower (Linear 129→64 + ReLU + Dropout).
            dropout:  Dropout probability applied after each tower and after the fusion ReLU.

        Creates:
            self.npp_tower:      Sequential NPP encoder, or None if use_npp is False.
            self.esm2_tower:     Sequential ESM2 encoder, or None if use_esm2 is False.
            self.ppi_tower:      Sequential PPI encoder, or None if use_ppi is False.
            self.fusion_linear:  Linear layer mapping concatenated tower outputs to 64 dims.
            self.fusion_relu:    ReLU applied after the fusion linear layer.
            self.fusion_dropout: Dropout applied before the final classifier.
            self.classifier:     Linear layer mapping the 64-dim embedding to 17 logits.

        Raises:
            ValueError: If all modalities are False.
        """
        super().__init__()

        if not use_npp and not use_esm2 and not use_ppi:
            raise ValueError("At least one input modality must be enabled.")

        self.use_npp = use_npp
        self.use_esm2 = use_esm2
        self.use_ppi = use_ppi

        if use_npp:
            self.npp_tower = nn.Sequential(
                nn.Linear(1905, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.npp_tower = None

        if use_esm2:
            self.esm2_tower = nn.Sequential(
                nn.Linear(1280, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.esm2_tower = None
            
        if use_ppi:
            self.ppi_tower = nn.Sequential(
                nn.Linear(129, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.ppi_tower = None

        fusion_dim = (64 if use_npp else 0) + (128 if use_esm2 else 0) + (64 if use_ppi else 0)
        self.fusion_linear = nn.Linear(fusion_dim, 64)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 17)

    def _fused_features(self, npp: torch.Tensor | None, esm2: torch.Tensor | None, ppi: torch.Tensor | None) -> torch.Tensor:
        """Pass each active modality through its tower and concatenate the results.

        Args:
            npp:  float32 tensor of shape (batch, 1905), or None.
            esm2: float32 tensor of shape (batch, 1280), or None.
            ppi:  float32 tensor of shape (batch, 129), or None.

        Returns:
            float32 tensor of shape (batch, fusion_dim).
        """
        features = []
        if self.use_npp and npp is not None:
            features.append(self.npp_tower(npp))
        if self.use_esm2 and esm2 is not None:
            features.append(self.esm2_tower(esm2))
        if self.use_ppi and ppi is not None:
            features.append(self.ppi_tower(ppi))
        return torch.cat(features, dim=1)

    def get_embedding(self, npp: torch.Tensor | None, esm2: torch.Tensor | None, ppi: torch.Tensor | None) -> torch.Tensor:
        """Return the 64-dimensional fusion representation before dropout and the classification head.

        Dropout is intentionally omitted here so that embeddings extracted at
        inference time are deterministic — repeated calls with the same input
        produce identical vectors, which is required for downstream clustering
        or visualisation of the learned gene representations.

        Args:
            npp:  float32 tensor of shape (batch, 1905), or None.
            esm2: float32 tensor of shape (batch, 1280), or None.
            ppi:  float32 tensor of shape (batch, 129), or None.

        Returns:
            float32 tensor of shape (batch, 64) — the post-ReLU fusion activation.
        """
        fused = self._fused_features(npp, esm2, ppi)
        return self.fusion_relu(self.fusion_linear(fused))

    def forward(self, npp: torch.Tensor | None, esm2: torch.Tensor | None, ppi: torch.Tensor | None) -> torch.Tensor:
        """Compute class logits for a batch of genes.

        Args:
            npp:  float32 tensor of shape (batch, 1905), or None.
            esm2: float32 tensor of shape (batch, 1280), or None.
            ppi:  float32 tensor of shape (batch, 129), or None.

        Returns:
            float32 tensor of shape (batch, 17) — unnormalised class scores
            (logits) for the 17 IRD phenotypic modules.
        """
        embedding = self.get_embedding(npp, esm2, ppi)
        # Apply dropout only in the forward pass (not in get_embedding) so that
        # embeddings extracted for downstream use remain deterministic.
        embedding = self.fusion_dropout(embedding)
        return self.classifier(embedding)
