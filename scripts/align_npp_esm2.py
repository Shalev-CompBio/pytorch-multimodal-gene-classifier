"""
================================================================================
align_npp_esm2.py — Stage 1c: Data Alignment (NPP x ESM2)
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
Inner-join the NPP matrix (~20,353 genes) with ESM2-650M embeddings (20,007 genes) 
on canonical gene symbol, then left-join module labels (442 labeled IRD genes). 
Produce two aligned output datasets: (1) a training set of genes with module labels 
(NPP + ESM2 + label), and (2) an unlabeled genome-wide set (NPP + ESM2 only). 
Both are saved as .npz arrays for downstream model training.

Inputs
------
- input/NPP_canonical_symbols_260626.csv
- input/esm2_650M_embeddings.npz
- input/gene_classification_20260412_1524.csv

Outputs
-------
- output/data_alignment/training_set_[timestamp].npz
- output/data_alignment/unlabeled_set_[timestamp].npz
- output/data_alignment/alignment_report_[timestamp].txt
- input/training_set.npz  [canonical copy]
- input/unlabeled_set.npz [canonical copy]

Usage
-----
python scripts/align_npp_esm2.py
================================================================================
"""

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

def save_and_report(path, **arrays):
    np.savez_compressed(path, **arrays)
    size_mb = path.stat().st_size / 1e6
    logging.info(f"Saved: {path} ({size_mb:.1f} MB)")


def main():
    # --- Configuration ---
    script_name = "data_alignment"
    
    ROOT_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = ROOT_DIR / "input"
    OUTPUT_DIR = ROOT_DIR / "output" / script_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    log_path = OUTPUT_DIR / f"run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s", 
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("Environment initialized. Ready for processing.")

    # ==========================================================================
    # Data Loading
    # ==========================================================================
    # --- Load NPP matrix ---
    npp_path = INPUT_DIR / "NPP_canonical_symbols_260626.csv"
    logging.info(f"Loading NPP matrix from {npp_path} ...")
    npp_df = pd.read_csv(npp_path, index_col=0)
    logging.info(f"NPP matrix loaded. Shape: {npp_df.shape}")
    print("\nNPP matrix head:")
    print(npp_df.iloc[:5, :5])

    # --- Deduplicate NPP index (keep first occurrence of each gene symbol) ---
    n_dup = npp_df.index.duplicated(keep='first').sum()
    if n_dup > 0:
        logging.warning(f"NPP index has {n_dup} duplicate entries — keeping first occurrence of each.")
        dup_names = npp_df.index[npp_df.index.duplicated(keep=False)].unique().tolist()
        logging.warning(f"Duplicated gene symbols: {sorted(dup_names)}")
        npp_df = npp_df[~npp_df.index.duplicated(keep='first')]
        logging.info(f"NPP after deduplication: {npp_df.shape}")
    else:
        logging.info("No duplicate NPP index entries found.")

    # --- Load ESM2 embeddings ---
    esm2_path = INPUT_DIR / "esm2_650M_embeddings.npz"
    logging.info(f"Loading ESM2 embeddings from {esm2_path} ...")
    esm2_archive = np.load(esm2_path, allow_pickle=True)
    esm2_genes = esm2_archive["gene_names"]       # shape: (20007,)
    esm2_embeddings = esm2_archive["embeddings"]  # shape: (20007, 1280)
    logging.info(f"ESM2 archive loaded. gene_names shape: {esm2_genes.shape}, embeddings shape: {esm2_embeddings.shape}")
    print(f"\nESM2 first 5 gene names: {esm2_genes[:5]}")
    print(f"ESM2 embedding sample (gene 0, dims 0–4): {esm2_embeddings[0, :5]}")

    # --- Load module labels ---
    labels_path = INPUT_DIR / "gene_classification_20260412_1524.csv"
    logging.info(f"Loading module labels from {labels_path} ...")
    labels_df = pd.read_csv(labels_path)
    logging.info(f"Module labels loaded. Shape: {labels_df.shape}")
    print("\nModule labels head:")
    print(labels_df.head())
    print(f"\nUnique modules: {sorted(labels_df['module_id'].unique())}")
    print(f"Total labeled genes: {len(labels_df)}")

    # ==========================================================================
    # Inner Join: NPP × ESM2
    # ==========================================================================
    # Build a reverse mapping from ESM2 gene name -> row index for O(1) lookup
    esm2_gene_to_idx = {gene: idx for idx, gene in enumerate(esm2_genes)}

    # Find intersection: NPP gene identifiers that also exist in ESM2
    npp_gene_list = npp_df.index.tolist()
    common_genes = [g for g in npp_gene_list if g in esm2_gene_to_idx]
    logging.info(f"Inner join result: {len(common_genes)} genes present in both NPP and ESM2")
    logging.info(f"  NPP-only genes (not in ESM2): {len(npp_gene_list) - len(common_genes)}")
    logging.info(f"  ESM2-only genes (not in NPP): {len(esm2_genes) - len(common_genes)}")

    # Subset and align both matrices to the common gene list
    aligned_npp = npp_df.loc[common_genes].values.astype(np.float32)  # (N_common, 1905)
    esm2_idx_order = [esm2_gene_to_idx[g] for g in common_genes]
    aligned_esm2 = esm2_embeddings[esm2_idx_order].astype(np.float32)  # (N_common, 1280)
    aligned_genes = np.array(common_genes)  # (N_common,)

    logging.info(f"aligned_npp shape: {aligned_npp.shape}")
    logging.info(f"aligned_esm2 shape: {aligned_esm2.shape}")
    assert len(aligned_genes) == aligned_npp.shape[0] == aligned_esm2.shape[0], \
        "Alignment mismatch: gene list length does not match matrix row counts."
    print(f"\nInner join complete. {len(aligned_genes)} genes retained.")

    # ==========================================================================
    # Left Join: Aligned Matrix + Module Labels
    # ==========================================================================
    aligned_df = pd.DataFrame({"gene": aligned_genes})

    aligned_df["gene_upper"] = aligned_df["gene"].str.upper()
    labels_upper = labels_df.copy()
    labels_upper["gene_upper"] = labels_upper["gene"].str.upper()
    aligned_df = aligned_df.merge(
        labels_upper[["gene_upper", "module_id", "module_qc_label", "stability_score", "classification"]],
        on="gene_upper",
        how="left"
    ).drop(columns="gene_upper")

    n_labeled_matched = aligned_df["module_id"].notna().sum()
    n_unlabeled = aligned_df["module_id"].isna().sum()
    logging.info("Left join complete.")
    logging.info(f"  Labeled genes matched in aligned matrix: {n_labeled_matched}")
    logging.info(f"  Unlabeled genes (NaN module): {n_unlabeled}")

    labels_gene_set_upper = {str(g).upper() for g in labels_df["gene"].tolist()}
    aligned_gene_set_upper = {str(g).upper() for g in aligned_genes.tolist()}
    missing_labeled_genes = labels_gene_set_upper - aligned_gene_set_upper

    if missing_labeled_genes:
        logging.warning(f"{len(missing_labeled_genes)} labeled IRD gene(s) NOT found in inner-join result:")
        for g in sorted(missing_labeled_genes):
            logging.warning(f"  - {g}")
    else:
        logging.info("All labeled IRD genes are present in the aligned matrix.")

    print(f"\nAligned DataFrame shape: {aligned_df.shape}")
    print(aligned_df[aligned_df["module_id"].notna()].head())

    # ==========================================================================
    # Split into Training Set and Unlabeled Set
    # ==========================================================================
    labeled_mask   = aligned_df["module_id"].notna().values
    unlabeled_mask = ~labeled_mask

    train_gene_names = aligned_genes[labeled_mask]   # (N_train,)
    train_npp        = aligned_npp[labeled_mask]      # (N_train, 1905)
    train_esm2       = aligned_esm2[labeled_mask]     # (N_train, 1280)
    train_labels     = aligned_df.loc[labeled_mask, "module_id"].values.astype(np.int32)  # (N_train,)

    unlabeled_gene_names = aligned_genes[unlabeled_mask]  # (N_unlabeled,)
    unlabeled_npp        = aligned_npp[unlabeled_mask]    # (N_unlabeled, 1905)
    unlabeled_esm2       = aligned_esm2[unlabeled_mask]   # (N_unlabeled, 1280)

    logging.info(f"Training set size:    {len(train_gene_names)} genes")
    logging.info(f"Unlabeled set size:   {len(unlabeled_gene_names)} genes")
    logging.info(f"train_npp shape:      {train_npp.shape}")
    logging.info(f"train_esm2 shape:     {train_esm2.shape}")
    logging.info(f"unlabeled_npp shape:  {unlabeled_npp.shape}")
    logging.info(f"unlabeled_esm2 shape: {unlabeled_esm2.shape}")

    assert len(train_gene_names) + len(unlabeled_gene_names) == len(aligned_genes), \
        "Set sizes do not add up to total aligned gene count."
    assert len(set(train_gene_names.tolist()) & set(unlabeled_gene_names.tolist())) == 0, \
        "Training set and unlabeled set overlap — this should never happen."
    print(f"\nTraining set: {len(train_gene_names)} genes | Unlabeled set: {len(unlabeled_gene_names)} genes")

    # ==========================================================================
    # Verification: All 442 Labeled Genes Present in Training Set
    # ==========================================================================
    train_gene_set = set(train_gene_names.tolist())
    expected_labeled_genes = set(labels_df["gene"].tolist())
    genes_not_recovered = expected_labeled_genes - train_gene_set

    all_input_genes = set(npp_gene_list) | set(esm2_genes.tolist())
    excluded_from_inputs = genes_not_recovered - all_input_genes
    join_bugs = genes_not_recovered & all_input_genes

    if excluded_from_inputs:
        logging.warning(
            f"{len(excluded_from_inputs)} labeled gene(s) absent from both NPP and ESM2 input files."
        )
        logging.warning("These are non-protein-coding loci excluded at Stage 1a (expected behavior):")
        for g in sorted(excluded_from_inputs):
            logging.warning(f"  - {g}")

    if join_bugs:
        logging.error(f"BUG: {len(join_bugs)} labeled gene(s) present in input data but LOST in join:")
        for g in sorted(join_bugs):
            logging.error(f"  - {g}")

    assert len(join_bugs) == 0, (
        f"Join bug detected: {len(join_bugs)} labeled gene(s) present in input data but not in training set: "
        f"{sorted(join_bugs)}"
    )

    n_labeled_in_training = len(train_gene_names)
    n_excluded = len(excluded_from_inputs)
    logging.info(
        f"Verification complete: {n_labeled_in_training}/{len(expected_labeled_genes)} labeled genes in training set. "
        f"{n_excluded} excluded (non-coding, absent from input data — expected)."
    )

    train_df_summary = aligned_df[labeled_mask].copy()
    module_counts = (
        train_df_summary
        .groupby(["module_id", "module_qc_label"])
        .size()
        .reset_index(name="gene_count")
        .sort_values("module_id")
    )
    print("\nPer-module gene counts in training set:")
    print(module_counts.to_string(index=False))
    print(f"\nTotal modules: {module_counts['module_id'].nunique()}")
    print(f"Gene count range: {module_counts['gene_count'].min()} to {module_counts['gene_count'].max()}")
    logging.info(
        f"Per-module count: {len(module_counts)} modules, "
        f"{module_counts['gene_count'].min()}-{module_counts['gene_count'].max()} genes each."
    )

    # ==========================================================================
    # Save Outputs
    # ==========================================================================
    ts_training_path  = OUTPUT_DIR / f"training_set_{timestamp}.npz"
    ts_unlabeled_path = OUTPUT_DIR / f"unlabeled_set_{timestamp}.npz"
    ts_report_path    = OUTPUT_DIR / f"alignment_report_{timestamp}.txt"

    save_and_report(
        ts_training_path,
        gene_names=train_gene_names,
        npp=train_npp,
        esm2=train_esm2,
        labels=train_labels
    )

    save_and_report(
        ts_unlabeled_path,
        gene_names=unlabeled_gene_names,
        npp=unlabeled_npp,
        esm2=unlabeled_esm2
    )

    canonical_training_path  = INPUT_DIR / "training_set.npz"
    canonical_unlabeled_path = INPUT_DIR / "unlabeled_set.npz"

    save_and_report(
        canonical_training_path,
        gene_names=train_gene_names,
        npp=train_npp,
        esm2=train_esm2,
        labels=train_labels
    )

    save_and_report(
        canonical_unlabeled_path,
        gene_names=unlabeled_gene_names,
        npp=unlabeled_npp,
        esm2=unlabeled_esm2
    )

    npp_gene_list_upper = {g.upper() for g in npp_gene_list}
    esm2_gene_list_upper = {g.upper() for g in esm2_genes.tolist()}
    all_input_genes_upper = npp_gene_list_upper | esm2_gene_list_upper
    labels_gene_upper = set(labels_df["gene"].str.upper())
    excluded_from_inputs_report = labels_gene_upper - all_input_genes_upper

    report_lines = [
        f"Data Alignment Report — {timestamp}",
        "=" * 50,
        "",
        "INPUT SUMMARY",
        f"  NPP matrix shape:             {npp_df.shape}",
        f"  ESM2 embeddings shape:        {esm2_embeddings.shape}",
        f"  Module labels (input):        {len(labels_df)} genes",
        "",
        "ALIGNMENT RESULTS",
        f"  Inner join (NPP and ESM2):    {len(aligned_genes)} genes retained",
        f"  NPP-only genes (dropped):     {len(npp_gene_list) - len(aligned_genes)}",
        f"  ESM2-only genes (dropped):    {len(esm2_genes) - len(aligned_genes)}",
        "",
        "SPLIT RESULT",
        f"  Training set (labeled):       {len(train_gene_names)} genes",
        f"  Unlabeled set:                {len(unlabeled_gene_names)} genes",
        "",
        "LABEL MATCHING",
        f"  Expected labeled genes:       {len(expected_labeled_genes)}",
        f"  Successfully matched:         {len(train_gene_names)}",
        f"  Missing labeled genes:        {len(excluded_from_inputs_report)}",
        f"  Excluded (non-coding):        {len(excluded_from_inputs_report)}",
        f"  Join bugs (code errors):      {len(join_bugs)}",
    ]

    if excluded_from_inputs_report:
        report_lines.append("")
        report_lines.append("LABELED GENES ABSENT FROM INPUTS (non-coding, excluded at Stage 1a):")
        for g in sorted(excluded_from_inputs_report):
            report_lines.append(f"  - {g}")

    if join_bugs:
        report_lines.append("")
        report_lines.append("LABELED GENES LOST TO JOIN BUGS:")
        for g in sorted(join_bugs):
            report_lines.append(f"  - {g}")

    report_lines += [
        "",
        "PER-MODULE GENE COUNTS (training set)",
        module_counts.to_string(index=False),
        "",
        "OUTPUT FILES",
        f"  {ts_training_path}",
        f"  {ts_unlabeled_path}",
        f"  {canonical_training_path}  [canonical]",
        f"  {canonical_unlabeled_path}  [canonical]",
    ]

    report_text = "\n".join(report_lines)
    with open(ts_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Alignment report written to: {ts_report_path}")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
