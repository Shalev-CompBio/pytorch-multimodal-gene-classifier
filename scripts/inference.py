"""
================================================================================
inference.py — Genome-wide inference for IRD Gene Classifier (Stage 4)
================================================================================
Author : Shalev Yaacov
Created: 2026-06-28
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
After cross-validation training, each fold-variant checkpoint has been optimised
on ~347 training genes. This script runs ensemble inference on ~20,000 unlabelled
human genes by averaging softmax probabilities across all N_FOLDS checkpoints for
each variant, which reduces per-fold variance and improves robustness. Three
variants are evaluated (npp_only, ppi_only, fusion_3tower). A multi-model
concordance analysis is then applied: genes where at least two variants agree on
the predicted module are retained and ranked by mean confidence, producing a
prioritised list of novel IRD gene hypotheses free of phenotypic annotation bias.

Inputs
------
- ../input/unlabeled_set_npp_esm2_ppi.npz: arrays gene_names (19573,),
  npp (19573, 1905) float32, esm2 (19573, 1280) float32, ppi (19573, 129) float32.
- ../checkpoints/{variant}_fold{N}_best.pt: trained state dictionaries for
  ThreeTowerClassifier, produced by train.py.

Outputs
-------
- ../output/inference/{variant}_genome_wide.csv: columns rank, gene_name,
  predicted_module, max_prob, ppi_coverage, prob_0 … prob_16.
- ../output/inference/concordant_candidates.csv: columns rank, gene_name,
  predicted_module, concordance_level, ppi_only_max_prob, npp_only_max_prob,
  fusion_3tower_max_prob, mean_confidence, rank_within_module, ppi_coverage,
  module_f1_cv, reliability_tier.

Usage
-----
    python scripts/inference.py
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import RANDOM_STATE, VARIANTS
from model import ThreeTowerClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "../checkpoints"
INFERENCE_OUTPUT_DIR = SCRIPT_DIR / "../output/inference"
INPUT_DATA_PATH = SCRIPT_DIR / "../input/unlabeled_set_npp_esm2_ppi.npz"

# Inference uses larger batches than training: the full unlabelled set fits in
# memory and there is no gradient overhead, so 512 is safe and reduces loop
# overhead without conflicting with the training BATCH_SIZE of 32.
INFERENCE_BATCH_SIZE = 512

N_FOLDS = 5
VARIANTS_TO_RUN = ["npp_only", "ppi_only", "fusion_3tower"]

# Per-module cross-validated F1 scores from ppi_only evaluate.py (Run 010).
# Used to assign reliability tiers to concordant candidate predictions.
MODULE_F1_CV = {
    0: 0.3736, 1: 0.3371, 2: 0.2941, 3: 0.3529, 4: 0.3103, 5: 0.7636,
    6: 0.4151, 7: 0.2264, 8: 0.4194, 9: 0.6522, 10: 0.6222, 11: 0.6400,
    12: 0.5556, 13: 0.3784, 14: 0.0741, 15: 0.3846, 16: 0.7619
}


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Fix random seeds for reproducibility.

    Args:
        seed: Integer seed value; defaults to the shared RANDOM_STATE constant
              from config.py so inference uses the same seed as training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def determine_reliability(f1_score: float) -> str:
    """Map a per-module cross-validated F1 score to a reliability tier label.

    Args:
        f1_score: Cross-validated F1 score for the predicted module, or NaN if
                  the module was not evaluated.

    Returns:
        "high"    if f1_score >= 0.55,
        "low"     if f1_score <  0.30,
        "medium"  otherwise,
        "unknown" if f1_score is NaN.
    """
    if pd.isna(f1_score):
        return "unknown"
    if f1_score >= 0.55:
        return "high"
    elif f1_score < 0.30:
        return "low"
    else:
        return "medium"


def run_inference() -> None:
    """Run genome-wide ensemble inference and concordance analysis.

    Stage 1 — Inference:
        For each variant in VARIANTS_TO_RUN, loads all N_FOLDS checkpoints,
        runs forward passes over the unlabelled gene set, averages the softmax
        probabilities across folds, and saves a ranked CSV.

    Stage 2 — Concordance analysis:
        Merges the per-variant predictions, assigns a concordance level to each
        gene (triple / ppi_npp / ppi_fusion / npp_fusion / none), filters out
        genes where all three variants disagree, ranks the remainder by mean
        confidence, and saves the final prioritised candidate list.
    """
    set_seeds()
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    print(f"Loading unlabeled data from {INPUT_DATA_PATH}...", flush=True)
    data = np.load(INPUT_DATA_PATH)
    gene_names = data["gene_names"]
    npp_data = torch.tensor(data["npp"], dtype=torch.float32)
    esm2_data = torch.tensor(data["esm2"], dtype=torch.float32)
    ppi_data = torch.tensor(data["ppi"], dtype=torch.float32)

    # Column 0 of ppi array contains boolean coverage flag (1=in STRING, 0=mean-imputed)
    ppi_coverage = ppi_data[:, 0].int().numpy()

    num_samples = len(gene_names)
    print(f"Total unlabeled genes: {num_samples}", flush=True)

    # --- Stage 1: Inference ---

    for variant_name in VARIANTS_TO_RUN:
        print(f"\nRunning inference for variant: {variant_name}", flush=True)
        flags = VARIANTS[variant_name]

        ensemble_probs = torch.zeros((num_samples, 17), dtype=torch.float32)

        for fold in range(1, N_FOLDS + 1):
            ckpt_path = CHECKPOINT_DIR / f"{variant_name}_fold{fold}_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")

            model = ThreeTowerClassifier(
                use_npp=flags["use_npp"],
                use_esm2=flags["use_esm2"],
                use_ppi=flags["use_ppi"]
            )

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            fold_probs = []
            with torch.no_grad():
                for i in range(0, num_samples, INFERENCE_BATCH_SIZE):
                    batch_npp = npp_data[i:i+INFERENCE_BATCH_SIZE]
                    batch_esm2 = esm2_data[i:i+INFERENCE_BATCH_SIZE]
                    batch_ppi = ppi_data[i:i+INFERENCE_BATCH_SIZE]

                    outputs = model(batch_npp, batch_esm2, batch_ppi)
                    probs = torch.softmax(outputs, dim=1)
                    fold_probs.append(probs)

            fold_probs = torch.cat(fold_probs, dim=0)
            ensemble_probs += fold_probs
            print(f"    fold {fold}/{N_FOLDS} done.", flush=True)

        # Ensembling probabilities across all 5 folds smooths variance and improves robustness
        ensemble_probs /= N_FOLDS

        max_probs, predicted_modules = torch.max(ensemble_probs, dim=1)

        ensemble_probs_np = ensemble_probs.numpy()
        max_probs_np = max_probs.numpy()
        predicted_modules_np = predicted_modules.numpy()

        # Constructing unranked DataFrame for current variant to prepare for CSV export
        df_dict = {
            "gene_name": gene_names,
            "predicted_module": predicted_modules_np,
            "max_prob": max_probs_np,
            "ppi_coverage": ppi_coverage
        }

        for i in range(17):
            df_dict[f"prob_{i}"] = ensemble_probs_np[:, i]

        df = pd.DataFrame(df_dict)

        # Rank genes globally by maximum predicted probability across any module
        df = df.sort_values(by="max_prob", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)

        output_csv = INFERENCE_OUTPUT_DIR / f"{variant_name}_genome_wide.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved {output_csv}", flush=True)

        print(f"{variant_name} summary:")
        print(f"    total genes:    {num_samples}")
        print(f"    mean max_prob:  {max_probs_np.mean():.4f}")
        print("    predicted_module distribution:")
        print(df["predicted_module"].value_counts().sort_index().to_string())

    # --- Stage 2: Concordance Analysis ---

    print("\nRunning concordance analysis...", flush=True)

    df_npp = pd.read_csv(INFERENCE_OUTPUT_DIR / "npp_only_genome_wide.csv")
    df_ppi = pd.read_csv(INFERENCE_OUTPUT_DIR / "ppi_only_genome_wide.csv")
    df_fusion = pd.read_csv(INFERENCE_OUTPUT_DIR / "fusion_3tower_genome_wide.csv")

    merged = df_ppi[['gene_name', 'predicted_module', 'max_prob', 'ppi_coverage']].rename(
        columns={'predicted_module': 'ppi_mod', 'max_prob': 'ppi_prob'}
    )

    merged = merged.merge(
        df_npp[['gene_name', 'predicted_module', 'max_prob']].rename(
            columns={'predicted_module': 'npp_mod', 'max_prob': 'npp_prob'}
        ),
        on='gene_name'
    )

    merged = merged.merge(
        df_fusion[['gene_name', 'predicted_module', 'max_prob']].rename(
            columns={'predicted_module': 'fusion_mod', 'max_prob': 'fusion_prob'}
        ),
        on='gene_name'
    )

    def get_concordance(row):
        """Return the concordance label for a row of merged variant predictions."""
        if row['npp_mod'] == row['ppi_mod'] == row['fusion_mod']:
            return "triple"
        elif row['ppi_mod'] == row['npp_mod']:
            return "ppi_npp"
        elif row['ppi_mod'] == row['fusion_mod']:
            return "ppi_fusion"
        elif row['npp_mod'] == row['fusion_mod']:
            return "npp_fusion"
        else:
            return "none"

    merged['concordance_level'] = merged.apply(get_concordance, axis=1)

    # Filter out genes where all three variants disagree (concordance_level == 'none')
    concordant = merged[merged['concordance_level'] != "none"].copy()

    def get_mean_confidence(row):
        """Return the mean max_prob across the variants that agree on the module."""
        lvl = row['concordance_level']
        if lvl == "triple":
            return (row['ppi_prob'] + row['npp_prob'] + row['fusion_prob']) / 3.0
        elif lvl == "ppi_npp":
            return (row['ppi_prob'] + row['npp_prob']) / 2.0
        elif lvl == "ppi_fusion":
            return (row['ppi_prob'] + row['fusion_prob']) / 2.0
        elif lvl == "npp_fusion":
            return (row['npp_prob'] + row['fusion_prob']) / 2.0
        return 0.0

    concordant['mean_confidence'] = concordant.apply(get_mean_confidence, axis=1)

    def get_final_module(row):
        """Return the agreed-upon predicted module index for a concordant gene."""
        if row['concordance_level'] == "npp_fusion":
            return row['npp_mod']
        else:
            return row['ppi_mod']

    concordant['predicted_module'] = concordant.apply(get_final_module, axis=1)

    concordant['module_f1_cv'] = concordant['predicted_module'].map(MODULE_F1_CV)
    concordant['reliability_tier'] = concordant['module_f1_cv'].apply(determine_reliability)

    # Rank all concordant genes globally to generate final prioritized candidate list
    concordant = concordant.sort_values(by='mean_confidence', ascending=False).reset_index(drop=True)
    concordant.insert(0, "rank", concordant.index + 1)

    # Calculate intra-module ranking to identify the strongest candidates for specific phenotypic groups
    concordant['rank_within_module'] = concordant.groupby('predicted_module')['mean_confidence'].rank(ascending=False, method='first').astype(int)

    # Rename probability columns to variant-specific names before final export
    concordant = concordant.rename(columns={
        "ppi_prob": "ppi_only_max_prob",
        "npp_prob": "npp_only_max_prob",
        "fusion_prob": "fusion_3tower_max_prob"
    })

    final_cols = [
        "rank", "gene_name", "predicted_module", "concordance_level",
        "ppi_only_max_prob", "npp_only_max_prob", "fusion_3tower_max_prob",
        "mean_confidence", "rank_within_module", "ppi_coverage",
        "module_f1_cv", "reliability_tier"
    ]

    concordant_final = concordant[final_cols]

    concordant_csv = INFERENCE_OUTPUT_DIR / "concordant_candidates.csv"
    concordant_final.to_csv(concordant_csv, index=False)

    print(f"\nConcordance analysis complete. Saved to {concordant_csv}", flush=True)
    print(f"Total concordant genes: {len(concordant_final)}")
    print("\nCount per concordance_level:")
    print(concordant_final['concordance_level'].value_counts().to_string())
    print("\nCount per reliability_tier:")
    print(concordant_final['reliability_tier'].value_counts().to_string())
    print("\nCount per predicted_module:")
    print(concordant_final['predicted_module'].value_counts().sort_index().to_string())


if __name__ == "__main__":
    run_inference()
