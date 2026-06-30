"""
02_build_edgelist.py
Filter STRING interactions and build a weighted gene-symbol edge list for node2vec.

Filtering steps (per chunk):
  1. combined_score >= 400
  2. Both proteins must be in string_to_gene.csv
  3. Remove self-loops (gene_a == gene_b)
  4. Normalize weight: combined_score / 1000.0
  5. Canonicalize: sort (gene_a, gene_b) alphabetically

Post-processing:
  6. Dedup by keeping max weight for each (gene_a, gene_b) pair
  7. Sort by gene_a, then gene_b

Outputs:
  ppi/processed/string_edgelist_400.tsv   -- tab-sep, no header: gene_a  gene_b  weight
  ppi/processed/edgelist_report.txt
"""

import pathlib
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent.parent
PPI_RAW  = ROOT / "ppi" / "raw"
PPI_PROC = ROOT / "ppi" / "processed"

LINKS_GZ  = PPI_RAW  / "9606.protein.links.v12.0.txt.gz"
MAPPING   = PPI_PROC / "string_to_gene.csv"
OUT_TSV   = PPI_PROC / "string_edgelist_400.tsv"
OUT_REPT  = PPI_PROC / "edgelist_report.txt"

SCORE_THRESH = 400
CHUNK_SIZE   = 500_000

# ── Load mapping ──────────────────────────────────────────────────────────────
print("Loading protein-to-gene mapping ...")
mapping_df = pd.read_csv(MAPPING)
pid_to_gene = dict(zip(mapping_df["string_protein_id"], mapping_df["gene_symbol"]))
print(f"  Proteins in mapping: {len(pid_to_gene):,}")

# ── Stream links file in chunks ───────────────────────────────────────────────
print(f"Processing links file in chunks of {CHUNK_SIZE:,} rows ...")

chunks_out = []

n_total         = 0
n_low_score     = 0
n_no_mapping    = 0
n_self_loop     = 0
n_passed        = 0

chunk_idx = 0
for chunk in pd.read_csv(LINKS_GZ, sep=" ", chunksize=CHUNK_SIZE):
    chunk_idx += 1
    n_total += len(chunk)

    # Step 1: score filter
    before_score = len(chunk)
    chunk = chunk[chunk["combined_score"] >= SCORE_THRESH]
    n_low_score += before_score - len(chunk)

    # Step 2: map to gene symbols
    chunk = chunk.copy()
    chunk["gene_a"] = chunk["protein1"].map(pid_to_gene)
    chunk["gene_b"] = chunk["protein2"].map(pid_to_gene)
    before_map = len(chunk)
    chunk = chunk.dropna(subset=["gene_a", "gene_b"])
    n_no_mapping += before_map - len(chunk)

    # Step 3: remove self-loops
    before_self = len(chunk)
    chunk = chunk[chunk["gene_a"] != chunk["gene_b"]]
    n_self_loop += before_self - len(chunk)

    # Step 4: normalize weight
    chunk["weight"] = (chunk["combined_score"] / 1000.0).round(3)

    # Step 5: canonicalize direction (sort gene_a < gene_b alphabetically)
    mask = chunk["gene_a"] > chunk["gene_b"]
    chunk.loc[mask, ["gene_a", "gene_b"]] = (
        chunk.loc[mask, ["gene_b", "gene_a"]].values
    )

    chunks_out.append(chunk[["gene_a", "gene_b", "weight"]])
    n_passed += len(chunk)

    if chunk_idx % 5 == 0:
        print(f"  ... processed chunk {chunk_idx}, total rows seen: {n_total:,}")

print(f"Done. Chunks processed: {chunk_idx}")

# ── Combine and deduplicate ───────────────────────────────────────────────────
print("Concatenating and deduplicating ...")
df = pd.concat(chunks_out, ignore_index=True)
before_dedup = len(df)

df = (
    df.groupby(["gene_a", "gene_b"], as_index=False)["weight"]
    .max()
)
n_dedup_removed = before_dedup - len(df)

# Step 7: sort
df = df.sort_values(["gene_a", "gene_b"]).reset_index(drop=True)

# ── Save TSV ──────────────────────────────────────────────────────────────────
df.to_csv(OUT_TSV, sep="\t", index=False, header=False, float_format="%.3f")
print(f"Saved edge list to: {OUT_TSV}")

# ── Statistics ────────────────────────────────────────────────────────────────
n_edges        = len(df)
unique_genes   = set(df["gene_a"]) | set(df["gene_b"])
n_unique_genes = len(unique_genes)
w_min  = df["weight"].min()
w_mean = df["weight"].mean()
w_max  = df["weight"].max()

report_lines = [
    "=== STRING Edge List Report ===",
    f"Score threshold               : {SCORE_THRESH}",
    f"Chunk size                    : {CHUNK_SIZE:,}",
    "",
    "--- Filtering steps ---",
    f"Total rows in links file      : {n_total:,}",
    f"Removed (score < {SCORE_THRESH})          : {n_low_score:,}",
    f"Removed (no gene mapping)     : {n_no_mapping:,}",
    f"Removed (self-loop)           : {n_self_loop:,}",
    f"Rows after per-chunk filter   : {n_passed:,}",
    f"Removed (dedup, kept max wt)  : {n_dedup_removed:,}",
    "",
    "--- Final edge list ---",
    f"Edges in final file           : {n_edges:,}",
    f"Unique genes in at least 1 edge : {n_unique_genes:,}",
    f"Min weight                    : {w_min:.3f}",
    f"Mean weight                   : {w_mean:.3f}",
    f"Max weight                    : {w_max:.3f}",
    "",
    f"Output: {OUT_TSV}",
]
report_str = "\n".join(report_lines)
print("\n" + report_str)
OUT_REPT.write_text(report_str, encoding="utf-8")
print(f"\nSaved report to: {OUT_REPT}")
