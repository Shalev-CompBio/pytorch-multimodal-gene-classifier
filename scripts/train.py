"""
================================================================================
train.py — 5-fold stratified cross-validation training for IRD module classifiers
================================================================================
Author : Shalev Yaacov
Created: 2026-06-26
Project: Multi-Modal IRD Gene Module Classifier
         (NPP × ESM2 × PPI three-tower fusion network)

Rationale
---------
With only 434 labelled genes and 17 imbalanced classes (10–63 genes per module),
a held-out test set would leave too few examples for reliable generalisation
estimates. Five-fold stratified cross-validation is therefore used: each gene
appears in exactly one validation fold, class proportions are preserved across
folds, and five independent model instances are trained and saved. The three
modality variants (fusion, npp_only, esm2_only) are trained in the same loop so
their checkpoints are directly comparable. Class weights are computed per fold
from training labels only to prevent any distribution leakage from the validation
fold into the loss. Early stopping on validation loss with a patience of 20 epochs
guards against overfitting on the small training sets (~347 samples per fold).

Inputs
------
- ../input/training_set.npz: gene_names, npp (434, 1905), esm2 (434, 1280),
  labels (434,) with values 0–16.

Outputs
-------
- ../checkpoints/{variant}_fold{k}_best.pt: state_dict and metadata for each fold.
- ../output/training/{variant}_cv_metrics.csv: per-fold macro-F1, weighted-F1,
  accuracy, and best epoch.
- Console: per-epoch loss and macro-F1, plus a cross-validation summary table.

Usage
-----
    python train.py
================================================================================
"""

from __future__ import annotations

import copy
import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from config import BATCH_SIZE, N_SPLITS, RANDOM_STATE, VARIANTS
from dataset import IRDDataset, compute_class_weights
from model import ThreeTowerClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "../input/training_set_npp_esm2_ppi.npz"
CHECKPOINT_DIR = SCRIPT_DIR / "../checkpoints"
TRAINING_OUTPUT_DIR = SCRIPT_DIR / "../output/training"

MAX_EPOCHS = 200
PATIENCE = 20


def set_seeds() -> None:
    """Fix all random seeds to make training runs fully reproducible.

    Sets seeds for PyTorch (CPU and all CUDA devices) and NumPy.
    Must be called before any random operations, including model
    initialisation and DataLoader shuffling.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one full pass over a DataLoader in either training or evaluation mode.

    The mode is inferred from whether an optimizer is provided: passing an
    optimizer enables gradient computation and parameter updates; omitting it
    runs the model in eval mode with gradients disabled.

    Args:
        model:     The classifier to run.
        loader:    DataLoader yielding (npp, esm2, labels) batches.
        criterion: Loss function (CrossEntropyLoss with class weights).
        device:    Compute device for tensors.
        optimizer: AdamW optimiser; pass None for validation/test passes.

    Returns:
        mean_loss: Sample-weighted mean loss over all batches.
        preds:     (N,) int array of predicted class indices.
        labels_np: (N,) int array of ground-truth class indices.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    for npp, esm2, ppi, labels in loader:
        npp_input = npp.to(device) if model.use_npp else None
        esm2_input = esm2.to(device) if model.use_esm2 else None
        ppi_input = ppi.to(device) if getattr(model, "use_ppi", False) else None
        labels = labels.to(device)

        if is_train:
            # set_to_none=True releases gradient memory rather than filling it
            # with zeros, reducing peak memory usage between steps.
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(npp_input, esm2_input, ppi_input)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        # Accumulate unnormalised loss so the final mean is sample-weighted
        # (not batch-count-weighted), which matters when the last batch is smaller.
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    mean_loss = total_loss / total_examples
    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    return mean_loss, preds, labels_np


def train_fold(
    dataset: IRDDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    variant: str,
    fold: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one model instance for a single cross-validation fold and save the best checkpoint.

    Args:
        dataset:   Full IRDDataset; train/val splits are created with Subset.
        train_idx: Indices of training genes for this fold.
        val_idx:   Indices of validation genes for this fold.
        variant:   One of 'fusion', 'npp_only', 'esm2_only'.
        fold:      1-based fold number (used in checkpoint filename and logging).
        device:    Compute device.

    Returns:
        dict with keys fold, macro_f1, weighted_f1, accuracy, best_epoch —
        all measured at the epoch of lowest validation loss.
    """
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ThreeTowerClassifier(**VARIANTS[variant]).to(device)
    # Class weights are computed from training labels only; using the full dataset
    # would leak the validation fold's label distribution into the loss, biasing
    class_weights = compute_class_weights(dataset.labels[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_macro_f1 = 0.0
    # deepcopy captures the full parameter tensor values at this moment; without
    # it, best_state would hold a reference to the live state_dict and would
    # silently track weight updates rather than freezing the best checkpoint.
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics = {"train_macro_f1": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "accuracy": 0.0}
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_preds, train_labels_np = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, device)

        train_macro_f1 = f1_score(train_labels_np, train_preds, average="macro", zero_division=0)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_accuracy = accuracy_score(val_labels, val_preds)

        print(
            f"{variant} fold {fold} | epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"train_macro_f1 {train_macro_f1:.4f} | val_macro_f1 {val_macro_f1:.4f}",
            flush=True,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            # Freeze a copy of the current weights as the new best checkpoint.
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = {
                "train_macro_f1": train_macro_f1,
                "macro_f1": val_macro_f1,
                "weighted_f1": val_weighted_f1,
                "accuracy": val_accuracy,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"{variant} fold {fold} | early stopping at epoch {epoch}", flush=True)
            break

    # Restore best weights before saving: the model's live weights may have
    # worsened after the best epoch, so load_state_dict is essential to ensure
    # the saved checkpoint matches the reported best metrics.
    model.load_state_dict(best_state)
    checkpoint_path = CHECKPOINT_DIR / f"{variant}_fold{fold}_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "fold": fold,
            "val_macro_f1": best_metrics["macro_f1"],
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    return {
        "fold": fold,
        "train_macro_f1": best_metrics["train_macro_f1"],
        "macro_f1": best_metrics["macro_f1"],
        "weighted_f1": best_metrics["weighted_f1"],
        "accuracy": best_metrics["accuracy"],
        "best_epoch": best_epoch,
    }


def write_metrics_csv(path: Path, metrics: list[dict[str, float | int]]) -> None:
    """Write per-fold cross-validation metrics to a CSV file.

    Args:
        path:    Destination file path (created or overwritten).
        metrics: List of dicts with keys fold, macro_f1, weighted_f1,
                 accuracy, best_epoch — one dict per fold.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "train_macro_f1", "macro_f1", "weighted_f1", "accuracy", "best_epoch"],
        )
        writer.writeheader()
        writer.writerows(metrics)


def print_variant_summary(variant: str, metrics: list[dict[str, float | int]]) -> None:
    """Print mean ± std (ddof=1) for macro-F1, weighted-F1, and accuracy across all folds.

    Args:
        variant: Variant name used as a header label.
        metrics: List of per-fold metric dicts produced by train_fold.
    """
    train_macro = np.array([row["train_macro_f1"] for row in metrics], dtype=float)
    macro = np.array([row["macro_f1"] for row in metrics], dtype=float)
    weighted = np.array([row["weighted_f1"] for row in metrics], dtype=float)
    accuracy = np.array([row["accuracy"] for row in metrics], dtype=float)

    print(f"\n{variant} cross-validation summary")
    # ddof=1 gives the sample standard deviation, appropriate for fold-level variance.
    print(f"train_macro_f1 {train_macro.mean():.4f} +/- {train_macro.std(ddof=1):.4f}")
    print(f"macro_f1       {macro.mean():.4f} +/- {macro.std(ddof=1):.4f}")
    print(f"weighted_f1    {weighted.mean():.4f} +/- {weighted.std(ddof=1):.4f}")
    print(f"accuracy       {accuracy.mean():.4f} +/- {accuracy.std(ddof=1):.4f}\n")


def main() -> None:
    """Orchestrate training of all variants across all folds and write summary outputs.

    Iterates over each variant in VARIANTS, then over each of the N_SPLITS folds,
    trains a model, saves a checkpoint, and collects metrics. After all folds for a
    variant are complete, writes a CSV and prints a summary. A final table compares
    all three variants side by side.
    """
    set_seeds()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = IRDDataset(str(INPUT_PATH))
    labels = dataset.labels
    # StratifiedKFold with shuffle=True and a fixed random_state ensures that
    # fold assignments are both class-balanced and reproducible across runs.
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_summaries = []
    for variant in VARIANTS:
        print(f"\nTraining variant: {variant}", flush=True)
        fold_metrics = []

        # Each call to splitter.split yields (train_idx, val_idx) for one fold.
        # The same splitter is reused across variants so all three models see
        # identical fold assignments, making their results directly comparable.
        for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
            metrics = train_fold(dataset, train_idx, val_idx, variant, fold, device)
            fold_metrics.append(metrics)

        write_metrics_csv(TRAINING_OUTPUT_DIR / f"{variant}_cv_metrics.csv", fold_metrics)
        print_variant_summary(variant, fold_metrics)

        all_summaries.append(
            {
                "variant": variant,
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_metrics])),
                "macro_f1_std": float(np.std([row["macro_f1"] for row in fold_metrics], ddof=1)),
                "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in fold_metrics])),
                "weighted_f1_std": float(np.std([row["weighted_f1"] for row in fold_metrics], ddof=1)),
                "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_metrics])),
                "accuracy_std": float(np.std([row["accuracy"] for row in fold_metrics], ddof=1)),
            }
        )

    print("\nSummary table")
    print("variant    macro_f1          weighted_f1       accuracy")
    for row in all_summaries:
        print(
            f"{row['variant']:<10} "
            f"{row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}  "
            f"{row['weighted_f1_mean']:.4f} +/- {row['weighted_f1_std']:.4f}  "
            f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )


if __name__ == "__main__":
    main()
