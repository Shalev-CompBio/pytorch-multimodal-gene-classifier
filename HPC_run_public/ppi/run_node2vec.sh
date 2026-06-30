#!/bin/bash
# ── SLURM directives — must appear before any executable lines ─────────────────
# Edit all placeholder values before submitting with sbatch.
# Note: SLURM reads these lines as literal text — shell variables are NOT
# expanded here, so all values must be written out directly.
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --partition=YOUR_PARTITION
#SBATCH --qos=normal
#SBATCH --job-name=ppi_node2vec
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/YOUR/LAB/ROOT/projects/ird_classifier/logs/node2vec_%j.out
#SBATCH --error=/YOUR/LAB/ROOT/projects/ird_classifier/logs/node2vec_%j.err

# ================================================================================
# run_node2vec.sh
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   SLURM batch job that runs node2vec (via pecanpy) on the STRING PPI edge list
#   to produce 128-dimensional graph embeddings for all genes in the network.
#   The resulting .emb file is then processed by assemble_ppi_embeddings.py to
#   align it to the master gene list and produce the final .npz input file.
#
# Inputs:
#   ${LAB_ROOT}/shared_data/ppi/string_edgelist_400.tsv
#     — tab-separated edge list: gene_a  gene_b  weight
#     — weight = STRING combined_score / 1000 (range 0.4–1.0)
#     — 879,158 edges covering 18,567 unique genes (STRING confidence >= 400)
#
# Outputs:
#   ${LAB_ROOT}/shared_data/ppi/ppi_node2vec_d128.emb
#     — word2vec text format: header line, then one gene vector per line
#
# Usage:
#   1. Edit the #SBATCH directives above: set account, partition, and log paths.
#   2. Set LAB_ROOT below (also update --output/--error paths above to match).
#   3. Ensure pecanpy is installed (run setup_ppi_environment.sh first).
#   4. sbatch run_node2vec.sh
#
# SLURM directive notes:
#   --account       : your SLURM account/allocation name
#   --partition     : a partition with at least 32 GB RAM and 16 CPU cores;
#                     GPU requested here only because the original partition
#                     required it — pecanpy is CPU-only and does not use the GPU
#   --cpus-per-task : parallelism for walk generation and word2vec training
#   --mem           : graph with ~18K nodes and ~880K edges fits comfortably in 32G
#   --time          : very conservative; actual runtime was ~39 seconds
#   --output/--error: edit to point to a writable log directory on your cluster
# ================================================================================

# ── Paths — set LAB_ROOT for your environment ──────────────────────────────────
# LAB_ROOT: top-level storage directory containing shared_data/, envs/, software/
# Also update the #SBATCH --output and --error paths above to match.
LAB_ROOT=/path/to/your/lab/root

# ── Environment setup ──────────────────────────────────────────────────────────
# Redirect temp files out of home directory (often quota-limited on HPC systems)
export TMPDIR=${LAB_ROOT}/tmp/conda_tmp
export PIP_CACHE_DIR=${LAB_ROOT}/tmp/pip_cache
export TORCH_HOME=${LAB_ROOT}/tmp/torch_cache

source ${LAB_ROOT}/software/miniforge3/etc/profile.d/conda.sh
conda activate ${LAB_ROOT}/envs/esm2

# ── Paths ──────────────────────────────────────────────────────────────────────
PPI_DATA=${LAB_ROOT}/shared_data/ppi

INPUT=${PPI_DATA}/string_edgelist_400.tsv
OUTPUT=${PPI_DATA}/ppi_node2vec_d128.emb

echo "=== node2vec via pecanpy ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Input     : $INPUT"
echo "Output    : $OUTPUT"
echo "Started   : $(date)"
echo ""

# ── Run pecanpy ───────────────────────────────────────────────────────────────
# node2vec parameters:
#   --dimensions 128  : embedding dimensionality — higher = more expressive, more memory
#   --walk-length 80  : steps per random walk — longer = more global context
#   --num-walks 10    : walks per node — more walks = better coverage, more compute
#   --p 1             : return parameter (BFS/DFS balance); p=1, q=1 = uniform random walk
#   --q 1             : in-out parameter; q<1 biases toward local community structure
#   --weighted        : use edge weights (STRING combined_score / 1000)
#   --workers 16      : must match --cpus-per-task in the SBATCH directives above
pecanpy \
    --input  $INPUT \
    --output $OUTPUT \
    --mode SparseOTF \
    --dimensions 128 \
    --walk-length 80 \
    --num-walks 10 \
    --p 1 \
    --q 1 \
    --weighted \
    --workers 16

# Note on mode: pecanpy may recommend PreCompFirstOrder over SparseOTF when
# p=1, q=1 (uniform walks). Both produce identical results; PreCompFirstOrder
# is faster for repeated runs but uses more memory during setup.

# ── Completion check ──────────────────────────────────────────────────────────
echo ""
echo "=== Job completed: $(date) ==="
echo "Output file line count (first line = header, rest = gene vectors):"
wc -l < $OUTPUT
echo ""
echo "First 3 lines of output (header + first 2 genes):"
head -3 $OUTPUT
echo ""
echo "Next step: python HPC_run_public/ppi/assemble_ppi_embeddings.py"
