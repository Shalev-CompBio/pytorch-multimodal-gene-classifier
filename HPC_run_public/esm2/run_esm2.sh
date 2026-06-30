#!/bin/bash
# ── SLURM directives — must appear before any executable lines ─────────────────
# Edit all placeholder values before submitting with sbatch.
# Note: SLURM reads these lines as literal text — shell variables are NOT
# expanded here, so all values must be written out directly.
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --qos=normal
#SBATCH --job-name=esm2_inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/YOUR/LAB/ROOT/projects/ird_classifier/logs/%j.out
#SBATCH --error=/YOUR/LAB/ROOT/projects/ird_classifier/logs/%j.err

# ================================================================================
# run_esm2.sh
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   SLURM batch submission script for the ESM2-650M inference job.
#   Runs esm2_inference.py on a GPU node to produce per-gene protein embeddings
#   for all ~20,007 genes in the dataset. This is a one-time compute step —
#   the resulting .npz file is downloaded locally and used for all training.
#
# Inputs:
#   esm2_inference.py (in the same directory)
#   ${LAB_ROOT}/shared_data/sequences/sequences_for_esm2.fasta
#
# Outputs:
#   ${LAB_ROOT}/shared_data/esm2_embeddings/esm2_650M_embeddings.npz
#
# Usage:
#   1. Edit the #SBATCH directives above: set account, partition, and log paths.
#   2. Set LAB_ROOT below (also update --output/--error paths above to match).
#   3. sbatch run_esm2.sh
#
# SLURM directive notes:
#   --account       : your SLURM account/allocation name
#   --partition     : GPU partition; must have GPU access and >=24 GB VRAM for ESM2-650M
#   --gres=gpu:1    : 1 GPU required; ESM2-650M needs ~24 GB VRAM
#   --cpus-per-task : CPU workers for data loading
#   --mem           : system RAM for sequence I/O and embedding storage
#   --time          : wall-time limit; inference takes ~15 min on an NVIDIA L40S
#   --output/--error: edit to point to a writable log directory on your cluster
# ================================================================================

# ── Paths — set LAB_ROOT for your environment ──────────────────────────────────
# LAB_ROOT: top-level directory containing shared_data/, envs/, and software/
# Also update the #SBATCH --output and --error paths above to match.
LAB_ROOT=/path/to/your/lab/root

# ── Environment setup ──────────────────────────────────────────────────────────
# Conda init script — path depends on where miniforge/anaconda is installed
source ${LAB_ROOT}/software/miniforge3/etc/profile.d/conda.sh

# Redirect temp files out of home (often quota-limited on HPC systems)
export TMPDIR=${LAB_ROOT}/tmp/conda_tmp
export PIP_CACHE_DIR=${LAB_ROOT}/tmp/pip_cache
# TORCH_HOME controls where PyTorch downloads model weights; pre-download
# ESM2 weights here before running to avoid timeout during job execution
export TORCH_HOME=${LAB_ROOT}/tmp/torch_cache

# Activate the conda environment that has fair-esm and PyTorch+CUDA installed
conda activate ${LAB_ROOT}/envs/esm2

# ── Run inference ──────────────────────────────────────────────────────────────
export LAB_ROOT=${LAB_ROOT}
python $(dirname "$0")/esm2_inference.py
