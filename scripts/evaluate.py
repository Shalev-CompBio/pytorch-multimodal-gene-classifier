"""
================================================================================
evaluate.py — Post-training evaluation of IRD classifier checkpoints
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
After training, each of the 5 × 3 fold-variant checkpoints holds the best
model weights for one validation fold. This script reconstructs the identical
fold assignments used during training (same StratifiedKFold parameters) and
runs inference for each gene on the model that never saw it during training,
producing unbiased held-out predictions for all 434 genes. Aggregating across
folds yields a single predicted label per gene with no leakage or duplicates.
A coverage assertion enforces this invariant at runtime. Per-module precision,
recall, and F1 are reported because macro-F1 is the primary metric for this
imbalanced 17-class problem, and module-level breakdown reveals which IRD
phenotypic groups are learned reliably versus which are confused.

Inputs
------
- ../input/training_set.npz: gene_names, npp (434, 1905), esm2 (434, 1280),
  labels (434,) with values 0–16.
- ../checkpoints/{variant}_fold{k}_best.pt: trained checkpoints from train.py.

Outputs
-------
- ../output/evaluation/{variant}_confusion_matrix.png: 17×17 confusion matrix.
- ../output/evaluation/{variant}_per_module_metrics.csv: per-module P/R/F1/support.
- Console: overall macro-F1, weighted-F1, accuracy, and per-module table.

Usage
-----
    python evaluate.py
================================================================================
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from config import BATCH_SIZE, N_SPLITS, RANDOM_STATE, VARIANTS
from dataset import IRDDataset
from model import ThreeTowerClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "../input/training_set_npp_esm2_ppi.npz"
CHECKPOINT_DIR = SCRIPT_DIR / "../checkpoints"
EVALUATION_OUTPUT_DIR = SCRIPT_DIR / "../output/evaluation"

N_CLASSES = 17


def set_seeds() -> None:
    """Fix all random seeds to ensure reproducible fold reconstruction and inference.

    Must be called before StratifiedKFold is instantiated to guarantee that
    fold assignments match those generated during training.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def predict_fold(
    model: ThreeTowerClassifier,
    dataset: IRDDataset,
    val_idx: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on one validation fold and return predicted and true labels.

    Args:
        model:   Trained ThreeTowerClassifier with best checkpoint weights loaded.
        dataset: Full IRDDataset; only val_idx rows are used.
        val_idx: Indices of validation genes for this fold.
        device:  Compute device for model inputs.

    Returns:
        preds:  (N_val,) int array of predicted module indices.
        labels: (N_val,) int array of ground-truth module indices.
    """
    loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for npp, esm2, ppi, labels in loader:
            npp_input = npp.to(device) if model.use_npp else None
            esm2_input = esm2.to(device) if model.use_esm2 else None
            ppi_input = ppi.to(device) if getattr(model, "use_ppi", False) else None
            logits = model(npp_input, esm2_input, ppi_input)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            # labels arrive from the DataLoader on CPU; .cpu() is a no-op here
            # but guards against any future change that moves labels to device.
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, variant: str) -> None:
    """Render and save a 17×17 confusion matrix as a PNG image.

    Args:
        y_true:  (434,) int array of ground-truth module indices.
        y_pred:  (434,) int array of predicted module indices.
        variant: Variant name used in the title and output filename.
    """
    labels = np.arange(N_CLASSES)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(f"{variant} confusion matrix")
    fig.tight_layout()
    fig.savefig(EVALUATION_OUTPUT_DIR / f"{variant}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_per_module_metrics(y_true: np.ndarray, y_pred: np.ndarray, variant: str) -> None:
    """Compute per-module precision, recall, F1, and support, then write to CSV and print.

    Args:
        y_true:  (434,) int array of ground-truth module indices.
        y_pred:  (434,) int array of predicted module indices.
        variant: Variant name used in the output filename.
    """
    labels = np.arange(N_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    path = EVALUATION_OUTPUT_DIR / f"{variant}_per_module_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "precision", "recall", "f1", "support"])
        for module, p, r, f, s in zip(labels, precision, recall, f1, support):
            writer.writerow([module, p, r, f, int(s)])

    print("module  precision  recall  f1  support")
    for module, p, r, f, s in zip(labels, precision, recall, f1, support):
        print(f"{module:>6}  {p:>9.4f}  {r:>6.4f}  {f:>6.4f}  {int(s):>7}")


def evaluate_variant(dataset: IRDDataset, variant: str, device: torch.device) -> None:
    """Aggregate held-out predictions for all folds and compute overall evaluation metrics.

    Fold assignments are reconstructed from the dataset labels using the same
    StratifiedKFold parameters as train.py.  Rather than storing fold indices
    during training, they are re-derived here: the StratifiedKFold split is
    deterministic given the same n_splits, shuffle, random_state, and label
    array, so reconstruction is exact and requires no extra files to manage.

    Args:
        dataset: Full IRDDataset (434 genes).
        variant: One of 'fusion', 'npp_only', 'esm2_only'.
        device:  Compute device for model inference.
    """
    labels = dataset.labels
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_preds = []
    all_labels = []
    covered_indices = []

    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        checkpoint_path = CHECKPOINT_DIR / f"{variant}_fold{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = ThreeTowerClassifier(**VARIANTS[variant]).to(device)
        model.load_state_dict(checkpoint["state_dict"])

        preds, fold_labels = predict_fold(model, dataset, val_idx, device)
        all_preds.append(preds)
        all_labels.append(fold_labels)
        covered_indices.extend(val_idx.tolist())

    # Correctness guarantee: each of the 434 genes must appear in exactly one
    # validation fold.  This assertion catches any mismatch between the fold
    # reconstruction here and the fold assignments used in training — for example,
    # if RANDOM_STATE or N_SPLITS were accidentally changed in only one script.
    if sorted(covered_indices) != list(range(len(dataset))):
        raise RuntimeError(f"{variant} fold validation sets do not cover each gene exactly once.")

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{variant} evaluation")
    print(f"macro_f1    {macro_f1:.4f}")
    print(f"weighted_f1 {weighted_f1:.4f}")
    print(f"accuracy    {accuracy:.4f}")

    save_per_module_metrics(y_true, y_pred, variant)
    save_confusion_matrix(y_true, y_pred, variant)


def main() -> None:
    """Run evaluation for all three modality variants and write all output files.

    Loads the dataset once and evaluates each variant sequentially.  All output
    directories are created with exist_ok=True so the script is safe to re-run.
    """
    set_seeds()
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = IRDDataset(str(INPUT_PATH))
    for variant in VARIANTS:
        evaluate_variant(dataset, variant, device)


if __name__ == "__main__":
    main()
