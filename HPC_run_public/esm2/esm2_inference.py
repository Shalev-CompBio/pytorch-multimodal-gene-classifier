#!/usr/bin/env python3
# ================================================================================
# esm2_inference.py
# Author  : Shalev Yaacov
# Created : June 2026 (as part of the IRD gene module classifier pipeline)
# Project : Multi-Modal IRD Gene Module Classifier
# ================================================================================
#
# Rationale:
#   ESM2-650M (Meta AI / FAIR) is run once as a GPU batch job on an HPC cluster
#   to produce per-gene protein sequence embeddings for all ~20,007 genes.
#   Mean-pooling over residue representations yields a fixed-length 1280-dim
#   vector per gene, which becomes the ESM2 input tower in the classifier.
#
# Inputs:
#   {LAB_ROOT}/shared_data/sequences/sequences_for_esm2.fasta
#     — FASTA file with one entry per gene, headers ">GENE_SYMBOL"
#     — 20,007 canonical human protein sequences
#
# Outputs:
#   {LAB_ROOT}/shared_data/esm2_embeddings/esm2_650M_embeddings.npz
#     — arrays: gene_names (20007,), embeddings (20007, 1280) float32
#
# Usage:
#   Set LAB_ROOT below, then:
#     python esm2_inference.py
#   Intended to be submitted via run_esm2.sh (SLURM). Requires a GPU.
# ================================================================================

import os
import time
import numpy as np
import torch
import esm
from pathlib import Path

# ── Configuration — set LAB_ROOT for your environment ──────────────────────────
# LAB_ROOT should be the top-level directory where shared_data/ and your project
# directory both live (e.g. your lab's scratch or project storage root).
LAB_ROOT = Path(os.environ.get("LAB_ROOT", "/path/to/your/lab/root"))
# ^ Set the LAB_ROOT environment variable before running, or edit this default.

FASTA_PATH  = LAB_ROOT / "shared_data/sequences/sequences_for_esm2.fasta"
OUTPUT_PATH = LAB_ROOT / "shared_data/esm2_embeddings/esm2_650M_embeddings.npz"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 1022  # ESM2 hard limit: 1024 tokens − 2 for BOS/EOS tokens
BATCH_SIZE  = 32    # sequences per GPU batch — reduce to 16 if OOM on smaller GPUs

# ── Parse FASTA ────────────────────────────────────────────────────────────────
def parse_fasta(path):
    seqs, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line)
    if name:
        seqs.append((name, "".join(buf)))
    return seqs

print("Parsing FASTA ...", flush=True)
sequences = parse_fasta(FASTA_PATH)
print(f"  {len(sequences)} sequences loaded", flush=True)

# Truncate sequences exceeding ESM2 context limit
n_truncated = sum(1 for _, s in sequences if len(s) > MAX_SEQ_LEN)
sequences = [(n, s[:MAX_SEQ_LEN]) for n, s in sequences]
print(f"  {n_truncated} sequences truncated to {MAX_SEQ_LEN} AA", flush=True)

# Sort by length — minimises padding waste within each batch
sequences.sort(key=lambda x: len(x[1]))

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading ESM2-650M ...", flush=True)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.cuda().eval()
print(f"  Model loaded on: {next(model.parameters()).device}", flush=True)

# ── Inference ──────────────────────────────────────────────────────────────────
all_names, all_embeds = [], []
t0 = time.time()

for i in range(0, len(sequences), BATCH_SIZE):
    batch = sequences[i : i + BATCH_SIZE]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.cuda()

    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)
        # Layer 33 is the final transformer layer of ESM2-650M (33 layers total)

    reps = out["representations"][33]  # (B, L+2, 1280)

    for j, (name, seq) in enumerate(batch):
        L = len(seq)
        # Positions 1..L are residue tokens; 0 = BOS, L+1 = EOS
        embed = reps[j, 1:L + 1].mean(0).cpu().float().numpy()
        all_names.append(name)
        all_embeds.append(embed)

    n_done  = min(i + BATCH_SIZE, len(sequences))
    elapsed = time.time() - t0
    rate    = n_done / elapsed
    eta     = (len(sequences) - n_done) / rate
    print(f"  {n_done}/{len(sequences)}  "
          f"{rate:.1f} seq/s  ETA {eta / 60:.0f} min", flush=True)

# ── Save ───────────────────────────────────────────────────────────────────────
embeddings_array = np.stack(all_embeds).astype(np.float32)  # (N, 1280)
names_array      = np.array(all_names)                       # (N,)

print(f"\nSaving to {OUTPUT_PATH} ...", flush=True)
np.savez_compressed(
    OUTPUT_PATH,
    gene_names = names_array,
    embeddings = embeddings_array,
)

total_min = (time.time() - t0) / 60
print(f"Done.  {len(all_names)} embeddings saved.  "
      f"Shape: {embeddings_array.shape}  "
      f"Total time: {total_min:.1f} min", flush=True)
