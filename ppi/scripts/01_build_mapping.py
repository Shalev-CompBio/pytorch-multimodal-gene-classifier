"""
01_build_mapping.py
Map each STRING protein ID (9606.ENSPXXXXXXXXXXX) to an uppercase HGNC gene
symbol from the master gene set (esm2_650M_embeddings.npz :: gene_names).

Three-pass strategy (stop at first match per protein):
  Pass 1 – direct preferred_name match (protein.info)
  Pass 2 – HGNC/BioMart alias match (protein.aliases)
  Pass 3 – UniProt accession bridge

Outputs:
  ppi/processed/string_to_gene.csv
  ppi/processed/mapping_report.txt
"""

import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PPI_RAW = ROOT / "ppi" / "raw"
PPI_PROC = ROOT / "ppi" / "processed"
PPI_PROC.mkdir(parents=True, exist_ok=True)

INFO_GZ    = PPI_RAW / "9606.protein.info.v12.0.txt.gz"
ALIASES_GZ = PPI_RAW / "9606.protein.aliases.v12.0.txt.gz"
ESM2_NPZ   = ROOT / "input" / "esm2_650M_embeddings.npz"

# ── Load master gene set ──────────────────────────────────────────────────────
print("Loading master gene set from ESM2 npz ...")
esm2 = np.load(ESM2_NPZ, allow_pickle=True)
master_genes = set(str(g).upper() for g in esm2["gene_names"])
print(f"  Master gene set size: {len(master_genes):,}")

# ── Load protein.info ─────────────────────────────────────────────────────────
print("Loading protein.info ...")
info = pd.read_csv(INFO_GZ, sep="\t", usecols=["#string_protein_id", "preferred_name"])
info.columns = ["string_protein_id", "preferred_name"]
total_proteins = len(info)
print(f"  Total STRING proteins: {total_proteins:,}")

# ── Pass 1: direct preferred_name match ──────────────────────────────────────
print("Pass 1: direct preferred_name match ...")
matched = {}  # string_protein_id -> (gene_symbol, match_type)

for _, row in info.iterrows():
    pid   = row["string_protein_id"]
    upper = str(row["preferred_name"]).upper()
    if upper in master_genes:
        matched[pid] = (upper, "direct")

pass1_count = len(matched)
print(f"  Pass 1 matched: {pass1_count:,}")

# ── Pass 2: HGNC alias match ─────────────────────────────────────────────────
print("Pass 2: HGNC/BioMart alias match ...")
ALIAS_SOURCES = {"HGNC", "BioMart_HUGO", "Gene_Name", "Ensembl_HGNC"}

aliases = pd.read_csv(ALIASES_GZ, sep="\t",
                      usecols=["#string_protein_id", "alias", "source"])
aliases.columns = ["string_protein_id", "alias", "source"]

# Filter to relevant sources
mask_src = aliases["source"].apply(
    lambda s: any(kw in str(s) for kw in ALIAS_SOURCES)
)
aliases_hgnc = aliases[mask_src].copy()

pass2_count = 0
for _, row in aliases_hgnc.iterrows():
    pid   = row["string_protein_id"]
    if pid in matched:
        continue
    upper = str(row["alias"]).upper()
    if upper in master_genes:
        matched[pid] = (upper, "alias")
        pass2_count += 1

print(f"  Pass 2 matched: {pass2_count:,}")

# ── Pass 3: UniProt accession bridge ─────────────────────────────────────────
print("Pass 3: UniProt accession bridge ...")
UNIPROT_SOURCES = {"UniProt_AC", "UniProt_SwissProt"}

mask_up = aliases["source"].apply(
    lambda s: any(kw in str(s) for kw in UNIPROT_SOURCES)
)
aliases_up = aliases[mask_up].copy()

pass3_count = 0
for _, row in aliases_up.iterrows():
    pid = row["string_protein_id"]
    if pid in matched:
        continue
    acc = str(row["alias"]).upper()
    if acc in master_genes:
        matched[pid] = (acc, "uniprot")
        pass3_count += 1

print(f"  Pass 3 matched: {pass3_count:,}")

# ── Build output DataFrame ────────────────────────────────────────────────────
records = [
    {"string_protein_id": pid, "gene_symbol": sym, "match_type": mtype}
    for pid, (sym, mtype) in matched.items()
]
mapping_df = pd.DataFrame(records)
mapping_df = mapping_df.sort_values("string_protein_id").reset_index(drop=True)

out_csv = PPI_PROC / "string_to_gene.csv"
mapping_df.to_csv(out_csv, index=False)
print(f"\nSaved mapping to: {out_csv}")

# ── Report ────────────────────────────────────────────────────────────────────
total_matched  = len(matched)
unique_symbols = mapping_df["gene_symbol"].nunique()
fraction       = unique_symbols / 20007  # master set size

report = [
    "=== STRING → Gene Symbol Mapping Report ===",
    f"Total STRING proteins in protein.info : {total_proteins:,}",
    f"Matched by pass 1 (direct name)       : {pass1_count:,}",
    f"Matched by pass 2 (HGNC alias)        : {pass2_count:,}",
    f"Matched by pass 3 (UniProt accession) : {pass3_count:,}",
    f"Total mapped proteins                 : {total_matched:,}",
    f"Unique gene symbols matched           : {unique_symbols:,}",
    f"Fraction of master gene set (20,007)  : {fraction:.4f} ({fraction*100:.1f}%)",
    "",
    "Note: Multiple STRING proteins may map to the same gene symbol (isoforms).",
    "Deduplication by max edge weight is performed in Step 3 (edgelist builder).",
]
report_str = "\n".join(report)
print("\n" + report_str)

out_report = PPI_PROC / "mapping_report.txt"
out_report.write_text(report_str)
print(f"\nSaved report to: {out_report}")
