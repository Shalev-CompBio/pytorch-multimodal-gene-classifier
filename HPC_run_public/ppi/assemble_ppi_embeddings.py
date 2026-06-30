#!/usr/bin/env python3
# ================================================================================
# assemble_ppi_embeddings.py
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   After run_node2vec.sh completes, the raw pecanpy .emb file contains vectors
#   only for genes present in the STRING network. This script aligns those vectors
#   to the full master gene list (20,007 genes, derived from the ESM2 embeddings),
#   imputes missing genes with the mean vector, and prepends a binary coverage flag
#   as column 0. The flag allows the classifier to distinguish true network
#   embeddings (flag=1) from imputed placeholders (flag=0) during training.
#
# Inputs:
#   ${LAB_ROOT}/shared_data/ppi/ppi_node2vec_d128.emb
#     — raw node2vec output; word2vec text format
#     — first line: "N_genes 128"; subsequent lines: "GENE f1 f2 ... f128"
#   ${LAB_ROOT}/shared_data/esm2_embeddings/esm2_650M_embeddings.npz
#     — used to extract the master gene list (gene_names array, shape (20007,))
#     — must already exist (produced by esm2_inference.py)
#
# Outputs:
#   ${LAB_ROOT}/shared_data/ppi_embeddings/ppi_node2vec_128.npz
#     — arrays:
#         gene_names  (20007,)      — gene symbols, identical order to ESM2 master
#         embeddings  (20007, 129)  — col 0 = coverage flag, cols 1-128 = node2vec
#
# Usage:
#   Set LAB_ROOT below, then run on the cluster gateway after run_node2vec.sh:
#     python assemble_ppi_embeddings.py
# ================================================================================

import os
import pathlib
import numpy as np

# ── Configuration — set LAB_ROOT for your environment ──────────────────────────
LAB_ROOT  = pathlib.Path(os.environ.get("LAB_ROOT", "/path/to/your/lab/root"))
# ^ Set the LAB_ROOT environment variable before running, or edit this default.

PPI_DATA  = LAB_ROOT / "shared_data" / "ppi"
PPI_EMB   = LAB_ROOT / "shared_data" / "ppi_embeddings"
ESM2_NPZ  = LAB_ROOT / "shared_data" / "esm2_embeddings" / "esm2_650M_embeddings.npz"
EMB_IN    = PPI_DATA / "ppi_node2vec_d128.emb"
EMB_OUT   = PPI_EMB  / "ppi_node2vec_128.npz"

PPI_EMB.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load node2vec .emb file ───────────────────────────────────────────
print("Loading node2vec embeddings ...")
n2v_dict = {}  # gene_symbol (as-is from file) -> np.array float32 shape (128,)

with open(EMB_IN, "r") as fh:
    header = fh.readline()  # "N_words dimensions" -- skip
    for line in fh:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        gene = parts[0]
        vec  = np.array(parts[1:], dtype=np.float32)
        n2v_dict[gene] = vec

# Build uppercase lookup to handle case differences between STRING and HGNC symbols
n2v_upper = {k.upper(): v for k, v in n2v_dict.items()}

n_emb_genes = len(n2v_dict)
print(f"  Gene vectors loaded from .emb file: {n_emb_genes:,}")
if n_emb_genes > 0:
    dim = len(next(iter(n2v_dict.values())))
    print(f"  Embedding dimension: {dim}")
    assert dim == 128, f"Expected 128-dim embeddings, got {dim}"

# ── Step 2: Load master gene list from ESM2 npz ───────────────────────────────
# The ESM2 gene list is the canonical master list — all downstream files
# (training_set.npz, unlabeled_set.npz) use this exact order.
print("Loading master gene list from ESM2 embeddings ...")
esm2 = np.load(ESM2_NPZ, allow_pickle=True)
gene_names = esm2["gene_names"]  # shape (20007,)
n_genes = len(gene_names)
print(f"  Master gene list size: {n_genes:,}")
assert n_genes == 20007, f"Expected 20,007 genes, got {n_genes}"

# ── Step 3: Align to master list ──────────────────────────────────────────────
print("Aligning node2vec vectors to master gene list ...")
DIM = 128
vectors = np.zeros((n_genes, DIM), dtype=np.float32)
flags   = np.zeros(n_genes, dtype=np.float32)  # 1.0 = covered by STRING, 0.0 = missing

for i, gene in enumerate(gene_names):
    gene_str = str(gene)
    # Try as-is first, then uppercase (STRING uses mixed-case gene symbols)
    if gene_str in n2v_dict:
        vectors[i] = n2v_dict[gene_str]
        flags[i] = 1.0
    elif gene_str.upper() in n2v_upper:
        vectors[i] = n2v_upper[gene_str.upper()]
        flags[i] = 1.0

n_covered = int(flags.sum())
n_missing = n_genes - n_covered
print(f"  Covered (flag=1): {n_covered:,} ({n_covered/n_genes*100:.1f}%)")
print(f"  Missing (flag=0): {n_missing:,}  ({n_missing/n_genes*100:.1f}%)")

# ── Step 4: Compute mean vector over covered genes ────────────────────────────
# The mean vector is used as the imputed value for genes absent from STRING.
# This is a neutral placeholder — the coverage flag (col 0 = 0) allows the
# classifier to learn to discount these imputed vectors during training.
print("Computing mean vector for imputation ...")
covered_mask = flags == 1.0
mean_vec = vectors[covered_mask].mean(axis=0)  # shape (128,)
print(f"  Mean vector norm: {np.linalg.norm(mean_vec):.4f}")

# ── Step 5: Impute missing genes with mean vector ─────────────────────────────
print("Imputing missing genes ...")
vectors[~covered_mask] = mean_vec

# ── Step 6: Build final matrix (20007, 129): col0=flag, cols1-128=embedding ───
# The flag is prepended as column 0 so the model can see it explicitly.
print("Building final matrix (20007, 129) ...")
flag_col   = flags.reshape(-1, 1)          # (20007, 1)
embeddings = np.concatenate([flag_col, vectors], axis=1)  # (20007, 129)
assert embeddings.shape == (20007, 129), f"Unexpected shape: {embeddings.shape}"
assert embeddings.dtype == np.float32

# Sanity: no NaN or Inf
assert not np.isnan(embeddings).any(),  "NaN values found in embeddings!"
assert not np.isinf(embeddings).any(),  "Inf values found in embeddings!"

# ── Step 7: Save npz ──────────────────────────────────────────────────────────
print(f"Saving to {EMB_OUT} ...")
np.savez(
    EMB_OUT,
    gene_names=gene_names,
    embeddings=embeddings,
)
print("  Saved.")

# ── Step 8: Summary ───────────────────────────────────────────────────────────
print("")
print("=== Assembly complete ===")
print(f"  Output file     : {EMB_OUT}")
print(f"  Shape           : {embeddings.shape}")
print(f"  Total genes     : {n_genes:,}")
print(f"  Covered (flag=1): {n_covered:,} ({n_covered/n_genes*100:.1f}%)")
print(f"  Imputed (flag=0): {n_missing:,}  ({n_missing/n_genes*100:.1f}%)")
print("")
print("Next step: python HPC_run_public/ppi/validate_ppi_embeddings.py")
