#!/bin/bash
# ================================================================================
# upload_to_cluster.sh
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   Run locally before any cluster work begins. Transfers the PPI edge list
#   (produced by ppi/scripts/02_build_edgelist.py) and all five cluster-side
#   scripts to the HPC cluster via scp through a jump host.
#   After this script completes, log in to the cluster and follow the printed
#   handoff checklist to complete the pipeline.
#
# Inputs (local — must exist before running):
#   ppi/processed/string_edgelist_400.tsv       — filtered STRING edge list (~16 MB)
#   HPC_run_public/ppi/setup_ppi_environment.sh
#   HPC_run_public/ppi/run_node2vec.sh
#   HPC_run_public/ppi/assemble_ppi_embeddings.py
#   HPC_run_public/ppi/validate_ppi_embeddings.py
#
# Outputs (remote, on cluster):
#   ${REMOTE_BASE}/shared_data/ppi/string_edgelist_400.tsv
#   ${REMOTE_BASE}/projects/ird_classifier/slurm/setup_ppi_environment.sh
#   ${REMOTE_BASE}/projects/ird_classifier/slurm/run_node2vec.sh
#   ${REMOTE_BASE}/projects/ird_classifier/scripts/assemble_ppi_embeddings.py
#   ${REMOTE_BASE}/projects/ird_classifier/scripts/validate_ppi_embeddings.py
#
# Usage:
#   1. Set JUMP, REMOTE, and REMOTE_BASE below.
#   2. Run from the project root: bash HPC_run_public/ppi/upload_to_cluster.sh
# ================================================================================

set -euo pipefail

# ── Configuration — set these for your cluster ─────────────────────────────────
# JUMP: your SSH jump host (bastion node), if required by your cluster.
# Remove the -J flag from scp commands below if a jump host is not needed.
JUMP=your_username@your.jump.host

# REMOTE: SSH address of the cluster login/gateway node
REMOTE=your_username@your.cluster.gateway

# REMOTE_BASE: top-level storage path on the cluster (equivalent to LAB_ROOT
# in the other scripts)
REMOTE_BASE=/path/to/your/lab/root

echo "=== Uploading PPI pipeline files to cluster ==="
echo "Jump host : $JUMP"
echo "Remote    : $REMOTE"
echo ""

# ── Edge list (largest file, upload first) ────────────────────────────────────
echo "[1/5] Uploading edge list (~16 MB) ..."
scp -J $JUMP \
    ppi/processed/string_edgelist_400.tsv \
    ${REMOTE}:${REMOTE_BASE}/shared_data/ppi/
echo "  Done."

# ── Cluster setup script ──────────────────────────────────────────────────────
echo "[2/5] Uploading setup_ppi_environment.sh ..."
scp -J $JUMP \
    HPC_run_public/ppi/setup_ppi_environment.sh \
    ${REMOTE}:${REMOTE_BASE}/projects/ird_classifier/slurm/
echo "  Done."

# ── SLURM node2vec job ────────────────────────────────────────────────────────
echo "[3/5] Uploading run_node2vec.sh ..."
scp -J $JUMP \
    HPC_run_public/ppi/run_node2vec.sh \
    ${REMOTE}:${REMOTE_BASE}/projects/ird_classifier/slurm/
echo "  Done."

# ── Assembly script ───────────────────────────────────────────────────────────
echo "[4/5] Uploading assemble_ppi_embeddings.py ..."
scp -J $JUMP \
    HPC_run_public/ppi/assemble_ppi_embeddings.py \
    ${REMOTE}:${REMOTE_BASE}/projects/ird_classifier/scripts/
echo "  Done."

# ── Validation script ─────────────────────────────────────────────────────────
echo "[5/5] Uploading validate_ppi_embeddings.py ..."
scp -J $JUMP \
    HPC_run_public/ppi/validate_ppi_embeddings.py \
    ${REMOTE}:${REMOTE_BASE}/projects/ird_classifier/scripts/
echo "  Done."

echo ""
echo "=== All files uploaded successfully ==="
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " HANDOFF CHECKLIST — run these steps on the cluster after SSH login"
echo "────────────────────────────────────────────────────────────────────────"
echo ""
echo "  1. Run setup (on cluster gateway — installs pecanpy, creates dirs + symlink):"
echo "       bash slurm/setup_ppi_environment.sh"
echo ""
echo "  2. Submit node2vec job (CPU-based, ~30-60 min estimated; actual ~39 sec on fast clusters):"
echo "       sbatch slurm/run_node2vec.sh"
echo ""
echo "  3. Monitor job progress:"
echo "       squeue -u \$USER"
echo ""
echo "  4. After job completes — assemble embeddings (run on gateway):"
echo "       python scripts/assemble_ppi_embeddings.py"
echo ""
echo "  5. Validate the assembled file:"
echo "       python scripts/validate_ppi_embeddings.py"
echo ""
echo "  Expected output: shared_data/ppi_embeddings/ppi_node2vec_128.npz"
echo "    shape (20007, 129), col0=coverage flag, cols1-128=node2vec dims"
echo "────────────────────────────────────────────────────────────────────────"
