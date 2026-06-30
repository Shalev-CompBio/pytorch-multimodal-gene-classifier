#!/bin/bash
# ================================================================================
# setup_ppi_environment.sh
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   One-time setup script to prepare the cluster environment for the PPI
#   node2vec pipeline. Run manually on the cluster gateway node (NOT via SLURM —
#   pecanpy installation requires network access which SLURM compute nodes
#   may not have). Creates shared_data directories, a symlink for the project's
#   input/, and installs pecanpy into the existing conda environment.
#
# Inputs:
#   An existing conda environment (ENV_PATH) with pip available.
#   The project input/ directory must already exist at ${PROJECT}/input/.
#
# Outputs:
#   ${LAB_ROOT}/shared_data/ppi/         — directory for edge list and .emb output
#   ${LAB_ROOT}/shared_data/ppi_embeddings/ — directory for assembled .npz
#   ${PROJECT}/input/ppi_embeddings      — symlink → shared_data/ppi_embeddings/
#   pecanpy installed in the conda env
#
# Usage:
#   1. Set LAB_ROOT, PROJECT, and ENV_PATH below.
#   2. bash setup_ppi_environment.sh
# ================================================================================

set -euo pipefail

# ── Configuration — set these for your environment ─────────────────────────────
# LAB_ROOT: top-level directory containing shared_data/, software/, envs/
LAB_ROOT=/path/to/your/lab/root

# PROJECT: full path to your local copy of the ird_classifier project
PROJECT=${LAB_ROOT}/projects/ird_classifier

# ENV_PATH: full path to the conda environment to install pecanpy into
# (must already have Python and pip; the ESM2 environment works for this)
ENV_PATH=${LAB_ROOT}/envs/esm2

# ── Environment setup ──────────────────────────────────────────────────────────
# Redirect temp files out of home directory (often quota-limited on HPC systems)
export TMPDIR=${LAB_ROOT}/tmp/conda_tmp
export PIP_CACHE_DIR=${LAB_ROOT}/tmp/pip_cache
export TORCH_HOME=${LAB_ROOT}/tmp/torch_cache

echo "=== PPI Setup Script ==="
echo "Base   : ${LAB_ROOT}"
echo "Project: ${PROJECT}"
echo ""

# ── Step 1: Create shared_data directories ────────────────────────────────────
echo "[1/4] Creating shared_data/ppi and shared_data/ppi_embeddings ..."
mkdir -p ${LAB_ROOT}/shared_data/ppi
mkdir -p ${LAB_ROOT}/shared_data/ppi_embeddings
echo "  Done."

# ── Step 2: Create symlink in project input/ ──────────────────────────────────
# The symlink lets assemble_ppi_embeddings.py write directly into shared_data/
# without needing to know its absolute path — the project refers to input/ppi_embeddings/
echo "[2/4] Creating symlink: ${PROJECT}/input/ppi_embeddings -> shared_data/ppi_embeddings ..."
SYMLINK=${PROJECT}/input/ppi_embeddings
TARGET=${LAB_ROOT}/shared_data/ppi_embeddings

if [ -L "${SYMLINK}" ]; then
    echo "  Symlink already exists at ${SYMLINK} — skipping."
elif [ -e "${SYMLINK}" ]; then
    echo "  WARNING: ${SYMLINK} exists and is not a symlink — manual inspection required."
else
    ln -s ${TARGET} ${SYMLINK}
    echo "  Symlink created: ${SYMLINK} -> ${TARGET}"
fi

# ── Step 3: Activate conda env and install pecanpy ────────────────────────────
echo "[3/4] Activating conda environment at ${ENV_PATH} ..."
source ${LAB_ROOT}/software/miniforge3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

echo "  Installing pecanpy ..."
pip install pecanpy
# pecanpy is the optimized node2vec implementation used for graph embedding.
# It uses numba JIT compilation and multi-core parallelism.

# ── Step 4: Verify installation ───────────────────────────────────────────────
echo "[4/4] Verifying pecanpy installation ..."
pecanpy --version
echo "  pecanpy installed successfully."

# ── Completion message ────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. The edge list should already be uploaded (via upload_to_cluster.sh)"
echo "  2. Submit node2vec job:"
echo "       sbatch ${PROJECT}/HPC_run_public/ppi/run_node2vec.sh"
echo "  3. Monitor:"
echo "       squeue -u \$USER"
echo "  4. After job completes, assemble embeddings (run on gateway):"
echo "       python ${PROJECT}/HPC_run_public/ppi/assemble_ppi_embeddings.py"
echo "  5. Validate:"
echo "       python ${PROJECT}/HPC_run_public/ppi/validate_ppi_embeddings.py"
