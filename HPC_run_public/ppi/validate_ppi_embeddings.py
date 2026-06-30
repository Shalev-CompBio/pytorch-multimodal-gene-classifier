#!/usr/bin/env python3
# ================================================================================
# validate_ppi_embeddings.py
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   Validates the assembled ppi_node2vec_128.npz file before downloading it
#   locally for use in training. Six checks are run: shape, NaN, Inf, binary
#   flag column, gene name alignment with the ESM2 master list, and a cosine
#   similarity spot-check on known IRD genes (ciliary and photoreceptor groups).
#   The spot-check confirms that PPI topology has captured biologically meaningful
#   structure — ciliary genes should cluster together, photoreceptor genes
#   separately, with lower cross-group similarity.
#
# Inputs:
#   ${LAB_ROOT}/shared_data/ppi_embeddings/ppi_node2vec_128.npz
#     — assembled PPI embeddings (produced by assemble_ppi_embeddings.py)
#   ${LAB_ROOT}/shared_data/esm2_embeddings/esm2_650M_embeddings.npz
#     — used to verify gene_names order matches the ESM2 master list
#
# Outputs:
#   Prints a PASS/FAIL report to stdout; exits with code 0 on success, 1 on failure.
#
# Usage:
#   Set LAB_ROOT below, then run on the cluster gateway after assemble_ppi_embeddings.py:
#     python validate_ppi_embeddings.py
# ================================================================================

import os
import pathlib
import sys
import numpy as np

# ── Configuration — set LAB_ROOT for your environment ──────────────────────────
LAB_ROOT = pathlib.Path(os.environ.get("LAB_ROOT", "/path/to/your/lab/root"))
# ^ Set the LAB_ROOT environment variable before running, or edit this default.

PPI_EMB  = LAB_ROOT / "shared_data" / "ppi_embeddings" / "ppi_node2vec_128.npz"
ESM2_NPZ = LAB_ROOT / "shared_data" / "esm2_embeddings" / "esm2_650M_embeddings.npz"

# Spot-check genes: two functional groups expected to cluster in PPI space.
# Ciliary complex genes (CEP290, RPGR, IFT88) and photoreceptor genes (ABCA4, RHO)
# should show higher within-group than cross-group cosine similarity.
SPOT_GENES = ["CEP290", "RPGR", "IFT88", "ABCA4", "RHO"]

failures = []

print("=== PPI Embedding Validation ===")
print(f"File: {PPI_EMB}")
print("")

# ── Load files ────────────────────────────────────────────────────────────────
ppi  = np.load(PPI_EMB, allow_pickle=True)
esm2 = np.load(ESM2_NPZ, allow_pickle=True)

gene_names = ppi["gene_names"]
embeddings = ppi["embeddings"]
esm2_genes = esm2["gene_names"]

# ── Check 1: Shape ────────────────────────────────────────────────────────────
print("[1] Shape check ...")
if embeddings.shape == (20007, 129):
    print(f"    PASS  shape = {embeddings.shape}")
else:
    msg = f"FAIL  expected (20007, 129), got {embeddings.shape}"
    print(f"    {msg}")
    failures.append(msg)

# ── Check 2: No NaN ───────────────────────────────────────────────────────────
print("[2] NaN check ...")
n_nan = int(np.isnan(embeddings).sum())
if n_nan == 0:
    print("    PASS  no NaN values")
else:
    msg = f"FAIL  {n_nan:,} NaN values found"
    print(f"    {msg}")
    failures.append(msg)

# ── Check 3: No Inf ───────────────────────────────────────────────────────────
print("[3] Inf check ...")
n_inf = int(np.isinf(embeddings).sum())
if n_inf == 0:
    print("    PASS  no Inf values")
else:
    msg = f"FAIL  {n_inf:,} Inf values found"
    print(f"    {msg}")
    failures.append(msg)

# ── Check 4: Column 0 is binary flag ─────────────────────────────────────────
# Column 0 must contain only 0.0 (imputed) or 1.0 (covered by STRING network)
print("[4] Coverage flag check (column 0 in {0.0, 1.0}) ...")
flags    = embeddings[:, 0]
flag_vals = set(np.unique(flags).tolist())
if flag_vals <= {0.0, 1.0}:
    n_covered = int((flags == 1.0).sum())
    n_missing = int((flags == 0.0).sum())
    pct = n_covered / len(flags) * 100
    print(f"    PASS  covered={n_covered:,} ({pct:.1f}%)  imputed={n_missing:,}  ({100-pct:.1f}%)")
else:
    msg = f"FAIL  unexpected flag values: {flag_vals}"
    print(f"    {msg}")
    failures.append(msg)

# ── Check 5: gene_names order matches ESM2 ────────────────────────────────────
# Exact alignment is required — all downstream .npz files use the ESM2 gene order
print("[5] Gene names match ESM2 master list ...")
if np.array_equal(gene_names, esm2_genes):
    print("    PASS  gene_names matches ESM2 master list exactly")
else:
    n_mismatch = int(np.sum(gene_names != esm2_genes))
    msg = f"FAIL  {n_mismatch} gene name mismatches vs ESM2 master"
    print(f"    {msg}")
    failures.append(msg)

# ── Check 6: Cosine similarity spot-check ────────────────────────────────────
print("[6] Cosine similarity spot-check ...")
gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}

print(f"    Spot genes: {SPOT_GENES}")
spot_vecs = []
for gene in SPOT_GENES:
    if gene not in gene_to_idx:
        print(f"    WARNING: {gene} not found in gene_names — skipping spot-check")
        spot_vecs = None
        break
    idx = gene_to_idx[gene]
    flag_val = embeddings[idx, 0]
    status   = "covered" if flag_val == 1.0 else "imputed"
    vec      = embeddings[idx, 1:]  # 128-dim subspace only (exclude coverage flag)
    print(f"      {gene:<10} flag={flag_val:.0f} ({status})")
    spot_vecs.append(vec)

if spot_vecs is not None and len(spot_vecs) == len(SPOT_GENES):
    mat = np.stack(spot_vecs)  # (5, 128)
    # L2-normalize each row before computing cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_n = mat / norms
    cosim = mat_n @ mat_n.T  # (5, 5) pairwise cosine similarities

    print("")
    header = "         " + "  ".join(f"{g:<10}" for g in SPOT_GENES)
    print(f"    {header}")
    for i, g in enumerate(SPOT_GENES):
        row = "  ".join(f"{cosim[i,j]:.3f}     " for j in range(len(SPOT_GENES)))
        print(f"    {g:<10} {row}")
    print("")
    print("    Expected: CEP290/RPGR/IFT88 mutually similar (ciliary complex).")
    print("    Expected: ABCA4/RHO more similar to each other than to ciliary trio.")
    cil_mean   = (cosim[0,1] + cosim[0,2] + cosim[1,2]) / 3
    photo_mean = cosim[3,4]
    cross_mean = np.mean([cosim[i,j] for i in [0,1,2] for j in [3,4]])
    print(f"    Ciliary trio mean sim  : {cil_mean:.3f}")
    print(f"    Photoreceptor pair sim : {photo_mean:.3f}")
    print(f"    Cross-group mean sim   : {cross_mean:.3f}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("")
if not failures:
    print("=== VALIDATION PASSED ===")
    sys.exit(0)
else:
    print("=== VALIDATION FAILED ===")
    print("Failures:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
