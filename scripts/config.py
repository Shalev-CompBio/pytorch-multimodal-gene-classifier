"""
================================================================================
config.py — Shared hyperparameters and experiment constants
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
Centralising all hyperparameters and experiment identifiers in one module
prevents the training and evaluation scripts from diverging silently. If
N_SPLITS or RANDOM_STATE were redefined independently in each file, a
one-character edit in one place would produce checkpoints trained on different
folds than those reconstructed during evaluation, invalidating all results.
Importing from a single source of truth makes such mismatches impossible and
makes grid-search or ablation sweeps trivially configurable.

Inputs
------
- None (this module defines constants only)

Outputs
-------
- None (imported by train.py and evaluate.py)

Usage
-----
Imported by train.py and evaluate.py — not run directly.
================================================================================
"""

# Three modality configurations: full fusion vs. each tower in isolation.
# Ablation variants allow us to quantify the marginal contribution of each
# input modality (NPP evolutionary profile, ESM2 embedding, and PPI embedding)
# to classification.
VARIANTS = {
    "npp_only":      {"use_npp": True,  "use_esm2": False, "use_ppi": False},
    "esm2_only":     {"use_npp": False, "use_esm2": True,  "use_ppi": False},
    "ppi_only":      {"use_npp": False, "use_esm2": False, "use_ppi": True},
    "npp_esm2":      {"use_npp": True,  "use_esm2": True,  "use_ppi": False},
    "npp_ppi":       {"use_npp": True,  "use_esm2": False, "use_ppi": True},
    "esm2_ppi":      {"use_npp": False, "use_esm2": True,  "use_ppi": True},
    "fusion_3tower": {"use_npp": True,  "use_esm2": True,  "use_ppi": True},
}

# Batch size of 32 balances gradient noise and GPU memory with ~347 training
# samples per fold; smaller batches would make loss estimates too noisy.
BATCH_SIZE = 32

# 5 folds gives an 80/20 train/validation split (~347/87 samples per fold),
# providing reliable generalisation estimates without leaving any single class
# with fewer than ~8 training examples.
N_SPLITS = 5

# Fixed random seed shared by StratifiedKFold in train.py and evaluate.py.
# Both scripts must use the same value to reconstruct identical fold assignments.
RANDOM_STATE = 42
