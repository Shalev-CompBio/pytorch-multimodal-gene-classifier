# HPC Run Scripts — Portable Version

This directory contains the cluster-side scripts from the
[Multi-Modal IRD Gene Module Classifier](../README.md) pipeline,
sanitized for public use. All institution-specific paths, account names,
and jump-host credentials have been replaced with clearly labeled variables.

> **This is the version to clone and adapt.** The original `HPC_run/`
> directory (not committed to the public repository) contains the exact
> paths used during the published run and is kept locally for
> reproducibility reference only.

---

## Prerequisites

- SLURM-based HPC cluster with at least one GPU partition (for ESM2 inference)
  and a multi-core CPU partition (for node2vec)
- Conda/Miniforge installed in your cluster environment
- A conda environment with PyTorch + CUDA and [fair-esm](https://github.com/facebookresearch/esm)
  (for `esm2/`) and with [pecanpy](https://github.com/krishnanlab/PecanPy) (for `ppi/`)

These scripts **will not run on a local machine** — they assume SLURM job
submission, GPU hardware, and cluster-resident data storage.

---

## Variables to Set Before Running

Each script defines these variables near the top with explanatory comments.
Set them once per script for your environment:

| Variable | Appears in | Description |
|---|---|---|
| `LAB_ROOT` | all scripts | Top-level storage directory containing `shared_data/`, `envs/`, and `software/` |
| `SLURM_ACCOUNT` | `run_esm2.sh`, `run_node2vec.sh` | Your SLURM account/allocation name (`#SBATCH --account=`) |
| `PARTITION` | `run_esm2.sh`, `run_node2vec.sh` | GPU or high-memory partition name (`#SBATCH --partition=`) |
| `ENV_PATH` | `setup_ppi_environment.sh` | Full path to the conda environment to install pecanpy into |
| `JUMP` | `upload_to_cluster.sh` | SSH jump host address (omit `-J` flag if not needed) |
| `REMOTE` | `upload_to_cluster.sh` | Cluster login/gateway node SSH address |
| `REMOTE_BASE` | `upload_to_cluster.sh` | Remote equivalent of `LAB_ROOT` |

`LAB_ROOT` can also be passed as an environment variable instead of editing
the scripts:
```bash
export LAB_ROOT=/your/storage/root
python assemble_ppi_embeddings.py
```

---

## Directory Structure

```
HPC_run_public/
├── README.md                     ← this file
├── esm2/                         ← ESM2 protein embedding pipeline (independent)
│   ├── esm2_inference.py         ← Python: runs ESM2-650M inference on GPU
│   └── run_esm2.sh               ← SLURM: submits esm2_inference.py as a batch job
└── ppi/                          ← PPI node2vec embedding pipeline
    ├── upload_to_cluster.sh      ← LOCAL: scp edge list + scripts to cluster
    ├── setup_ppi_environment.sh  ← CLUSTER GATEWAY: installs pecanpy, creates dirs/symlinks
    ├── run_node2vec.sh           ← SLURM: submits node2vec (pecanpy) as a batch job
    ├── assemble_ppi_embeddings.py← CLUSTER GATEWAY: aligns node2vec output to master gene list
    └── validate_ppi_embeddings.py← CLUSTER GATEWAY: validates shape, flags, and biological structure
```

---

## Execution Order

### `esm2/` — ESM2 Protein Embeddings (independent pipeline)

This pipeline is independent of `ppi/` and can run first or in parallel.

1. **Set variables** in `run_esm2.sh` (`LAB_ROOT`, `SLURM_ACCOUNT`, `PARTITION`) and in
   `esm2_inference.py` (`LAB_ROOT`)
2. **Pre-download ESM2 model weights** to `${LAB_ROOT}/tmp/torch_cache/` to avoid
   timeout during the job (weights are ~2.4 GB for ESM2-650M)
3. **Submit the job** from the cluster:
   ```bash
   sbatch run_esm2.sh
   ```
4. **Download the output** locally after the job completes:
   ```
   ${LAB_ROOT}/shared_data/esm2_embeddings/esm2_650M_embeddings.npz
   ```
   Copy to `input/esm2_650M_embeddings.npz` in your local project directory.

### `ppi/` — PPI node2vec Embeddings (5-step pipeline)

Run from your **local machine** first, then complete steps on the cluster:

1. **[LOCAL]** Build the edge list locally (using `ppi/scripts/01_build_mapping.py`
   and `ppi/scripts/02_build_edgelist.py`) — produces `ppi/processed/string_edgelist_400.tsv`
2. **[LOCAL]** Set variables in `upload_to_cluster.sh` (`JUMP`, `REMOTE`, `REMOTE_BASE`)
   and run from the project root:
   ```bash
   bash HPC_run_public/ppi/upload_to_cluster.sh
   ```
   This transfers the edge list and all four cluster scripts to the cluster.
3. **[CLUSTER GATEWAY]** Set variables in `setup_ppi_environment.sh` (`LAB_ROOT`, `ENV_PATH`)
   and run manually (not via SLURM — requires network for pip):
   ```bash
   bash setup_ppi_environment.sh
   ```
4. **[SLURM]** Set variables in `run_node2vec.sh` (`LAB_ROOT`, `SLURM_ACCOUNT`, `PARTITION`)
   and submit:
   ```bash
   sbatch run_node2vec.sh
   ```
5. **[CLUSTER GATEWAY]** After the job completes, set `LAB_ROOT` in
   `assemble_ppi_embeddings.py` and run:
   ```bash
   python assemble_ppi_embeddings.py
   ```
6. **[CLUSTER GATEWAY]** Set `LAB_ROOT` in `validate_ppi_embeddings.py` and run:
   ```bash
   python validate_ppi_embeddings.py
   ```
7. **[LOCAL]** Download the validated output:
   ```
   ${LAB_ROOT}/shared_data/ppi_embeddings/ppi_node2vec_128.npz
   ```
   Copy to `input/ppi_node2vec_128.npz` in your local project directory.

---

## Expected Outputs

| File | Shape | Description |
|---|---|---|
| `esm2_650M_embeddings.npz` | (20007, 1280) float32 | ESM2-650M mean-pooled protein embeddings |
| `ppi_node2vec_128.npz` | (20007, 129) float32 | node2vec PPI embeddings; col 0 = STRING coverage flag |

Both files use the same gene order (derived from the ESM2 master gene list).
The `ppi_node2vec_128.npz` coverage flag encodes whether a gene was present
in the STRING network (flag=1) or received mean-vector imputation (flag=0).

---

## Notes on Compute Requirements

- **ESM2 inference**: requires a GPU with ≥24 GB VRAM (tested on NVIDIA L40S, 48 GB).
  With `BATCH_SIZE=32` and sequences sorted by length, inference over ~20,007 genes
  takes approximately 13–15 minutes.
- **node2vec (pecanpy)**: CPU-only despite being submitted to a GPU partition
  in the original run. 16 cores, 32 GB RAM. The graph (~18,500 nodes, ~880,000
  edges) completes in under 1 minute with pecanpy's JIT-compiled implementation.
