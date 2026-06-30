"""
================================================================================
append_ppi_embeddings.py — Aligns PPI embeddings to training and unlabeled sets
================================================================================
Author : Shalev Yaacov
Created: 2026-06-28
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
Integrates Protein-Protein Interaction (PPI) node2vec 128D embeddings into the
feature sets (training and unlabeled) to serve as a biological context. Genes
missing from the PPI network are identified but handled downstream (e.g. through
imputation vectors encoded with flag=0). Ensures identical ordering between
original data and aligned structural/sequence information.

Inputs
------
- input/ppi_node2vec_128.npz: gene_names (20007,), embeddings (20007, 129)
- input/training_set.npz: gene_names (434,), npp, esm2, labels
- input/unlabeled_set.npz: gene_names (19573,), npp, esm2

Outputs
-------
- input/training_set_npp_esm2_ppi.npz: original arrays + ppi (434, 129)
- input/unlabeled_set_npp_esm2_ppi.npz: original arrays + ppi (19573, 129)

Usage
-----
python scripts/align_ppi_to_datasets.py
================================================================================
"""

import numpy as np

# Constants
PPI_PATH = 'input/ppi_node2vec_128.npz'
TRAINING_PATH = 'input/training_set.npz'
UNLABELED_PATH = 'input/unlabeled_set.npz'
TRAINING_OUT_PATH = 'input/training_set_npp_esm2_ppi.npz'
UNLABELED_OUT_PATH = 'input/unlabeled_set_npp_esm2_ppi.npz'

def process_dataset(in_path, out_path, dataset_name, ppi_gene_to_idx, ppi_embeddings):
    print(f"[{dataset_name.upper()}_LOAD] Loading original data from {in_path}...")
    data = np.load(in_path, allow_pickle=True)
    genes = data['gene_names']
    
    # We construct the PPI features array directly aligned with the dataset gene order.
    ppi = []
    for gene in genes:
        if gene not in ppi_gene_to_idx:
            raise ValueError(f"Gene '{gene}' in {dataset_name} not found in PPI data.")
        ppi.append(ppi_embeddings[ppi_gene_to_idx[gene]])
    ppi = np.array(ppi)
    
    # Repackage the data with the new aligned modality.
    save_dict = {k: data[k] for k in data.files}
    save_dict['ppi'] = ppi
    
    print(f"[{dataset_name.upper()}_SAVE] Saving updated data to {out_path}...")
    np.savez(out_path, **save_dict)
    
    return ppi, genes

def main():
    # --- Stage 1: Load PPI Embeddings ---
    print(f"[PPI_LOAD] Loading PPI vectors from {PPI_PATH}...")
    ppi_data = np.load(PPI_PATH, allow_pickle=True)
    ppi_gene_names = ppi_data['gene_names']
    ppi_embeddings = ppi_data['embeddings']
    
    # Map gene identifiers to their row index for O(1) alignment lookups.
    ppi_gene_to_idx = {gene: idx for idx, gene in enumerate(ppi_gene_names)}
    
    # --- Stage 2: Process Datasets ---
    train_ppi, train_genes = process_dataset(
        TRAINING_PATH, TRAINING_OUT_PATH, "training", ppi_gene_to_idx, ppi_embeddings
    )
    unl_ppi, unl_genes = process_dataset(
        UNLABELED_PATH, UNLABELED_OUT_PATH, "unlabeled", ppi_gene_to_idx, ppi_embeddings
    )
    
    # --- Stage 3: Reporting ---
    print("\n[REPORT] Validating shapes and statistics...")
    print(f"1. Shape of ppi array added to training_set: {train_ppi.shape}")
    print(f"2. Shape of ppi array added to unlabeled_set: {unl_ppi.shape}")
    
    flags = train_ppi[:, 0]
    num_flag_1 = np.sum(flags == 1)
    frac_flag_1 = num_flag_1 / len(flags)
    num_flag_0 = np.sum(flags == 0)
    frac_flag_0 = num_flag_0 / len(flags)
    
    print(f"3. Number of training genes with coverage flag=1: {num_flag_1} (Fraction: {frac_flag_1:.4f})")
    print(f"4. Number of training genes with coverage flag=0: {num_flag_0} (Fraction: {frac_flag_0:.4f})")
    
    train_nan = np.isnan(train_ppi).any()
    train_inf = np.isinf(train_ppi).any()
    unl_nan = np.isnan(unl_ppi).any()
    unl_inf = np.isinf(unl_ppi).any()
    print("5. Confirm no NaN or Inf:")
    print(f"   training_set ppi array has NaN: {train_nan}, Inf: {train_inf}")
    print(f"   unlabeled_set ppi array has NaN: {unl_nan}, Inf: {unl_inf}")
    
    print("6. 3 sample rows from training_set ppi array:")
    for i in range(min(3, len(train_genes))):
        print(f"   Gene: {train_genes[i]}, Flag: {train_ppi[i, 0]}, First 3 dims: {train_ppi[i, 1:4]}")

if __name__ == '__main__':
    main()
