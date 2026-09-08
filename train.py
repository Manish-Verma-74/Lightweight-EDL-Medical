"""
Train an EDL-head, R-EDL-head, or softmax backbone on HAM10000.

Supported loss functions:
    - edl
    - redl
    - softmax

R-EDL:
    Chen, Gao, Xu, ICLR 2024
    "R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning"

Experiment design:
    - HAM10000
    - Lesion-level StratifiedGroupKFold split
    - WeightedRandomSampler for class imbalance
    - EfficientNet-B0 / MobileNetV3-Small / ShuffleNetV2
    - Standard augmentation currently handled by datasets.ham10000
    - Accuracy + ECE
    - Best checkpoint selected using validation accuracy

Example R-EDL run:

    python train.py \
        --backbone efficientnet_b0 \
        --loss_fn redl \
        --redl_lambda 0.1 \
        --dataset ham10000 \
        --augmentation standard \
        --seed 42 \
        --data_root data/ham10000 \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --annealing_step 10 \
        --patience 5 \
        --checkpoint_dir checkpoints \
        --log_path results/framework_screening.csv
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold

from datasets.ham10000 import (
    HAM10000Dataset,
    default_transforms,
    CLASS_NAMES,
)

from losses.evidential_loss import (
    edl_mse_loss,
    edl_predictions,
    redl_loss,
    redl_predictions,
)

from metrics.ece import compute_ece

from models.backbone_factory import get_backbone


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Train EDL, R-EDL, or Softmax model on HAM10000."
    )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    p.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=[
            "efficientnet_b0",
            "mobilenet_v3_small",
            "shufflenet_v2",
        ],
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    p.add_argument(
        "--loss_fn",
        default="edl",
        choices=[
            "edl",
            "redl",
            "softmax",
        ],
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    p.add_argument(
        "--dataset",
        default="ham10000",
    )

    # --------------------------------------------------------
    # Augmentation label
    # --------------------------------------------------------

    p.add_argument(
        "--augmentation",
        default="standard",
        help=(
            "Label for the augmentation strategy used "
            "(e.g. standard, mixup, cutmix, randaugment). "
            "Actual augmentation logic is implemented "
            "in datasets/*.py."
        ),
    )

    # --------------------------------------------------------
    # Dataset root
    # --------------------------------------------------------

    p.add_argument(
        "--data_root",
        default="data/ham10000",
    )

    # --------------------------------------------------------
    # Training parameters
    # --------------------------------------------------------

    p.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--annealing_step",
        type=int,
        default=10,
    )

    # --------------------------------------------------------
    # R-EDL lambda
    # --------------------------------------------------------

    p.add_argument(
        "--redl_lambda",
        type=float,
        default=0.1,
        help=(
            "Prior weight hyperparameter for R-EDL. "
            "Used only when --loss_fn redl."
        ),
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    p.add_argument(
        "--val_split",
        type=float,
        default=0.15,
    )

    # --------------------------------------------------------
    # Debug subset
    # --------------------------------------------------------

    p.add_argument(
        "--debug_subset",
        type=int,
        default=None,
        help=(
            "If set, train using only N samples "
            "for a fast sanity check."
        ),
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory where experiment checkpoints are stored.",
    )

    # --------------------------------------------------------
    # Results log
    # --------------------------------------------------------

    p.add_argument(
        "--log_path",
        default="results/master_log.csv",
    )

    return p.parse_args()


# ============================================================
# Checkpoint identity
# ============================================================

def checkpoint_path(
    checkpoint_dir,
    dataset,
    backbone,
    loss_fn,
    augmentation,
    seed,
    debug_subset=None,
    redl_lambda=0.1,
):
    """
    Generate a unique checkpoint filename.

    Standard EDL / Softmax:

        ham10000_efficientnet_b0_edl_standard_seed42.pt

    R-EDL:

        ham10000_efficientnet_b0_redl_standard_seed42_lam0.1.pt

    Debug run:

        ham10000_efficientnet_b0_redl_standard_seed42_lam0.1_debug50.pt
    """

    debug_suffix = (
        f"_debug{debug_subset}"
        if debug_subset is not None
        else ""
    )

    lambda_suffix = (
        f"_lam{redl_lambda:g}"
        if loss_fn == "redl"
        else ""
    )

    fname = (
        f"{dataset}_"
        f"{backbone}_"
        f"{loss_fn}_"
        f"{augmentation}_"
        f"seed{seed}"
        f"{lambda_suffix}"
        f"{debug_suffix}.pt"
    )

    return os.path.join(
        checkpoint_dir,
        fname,
    )


def best_checkpoint_path(ckpt_path):
    """
    Convert latest checkpoint path into best checkpoint path.
    """

    root, ext = os.path.splitext(
        ckpt_path
    )

    return f"{root}_best{ext}"


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    args,
    best_metric,
    best_epoch,
):
    """
    Save:

        - model state
        - optimizer state
        - experiment configuration
        - hyperparameters
        - best validation information
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    torch.save(
        {
            # ------------------------------------------------
            # Training state
            # ------------------------------------------------

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            # ------------------------------------------------
            # Experiment identity
            # ------------------------------------------------

            "dataset":
                args.dataset,

            "backbone":
                args.backbone,

            "loss_fn":
                args.loss_fn,

            "augmentation":
                args.augmentation,

            "seed":
                args.seed,

            # ------------------------------------------------
            # Hyperparameters
            # ------------------------------------------------

            "batch_size":
                args.batch_size,

            "lr":
                args.lr,

            "annealing_step":
                args.annealing_step,

            "redl_lambda":
                args.redl_lambda,

            "val_split":
                args.val_split,

            # ------------------------------------------------
            # Best validation result
            # ------------------------------------------------

            "best_metric":
                best_metric,

            "best_epoch":
                best_epoch,
        },
        path,
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint_if_exists(
    path,
    model,
    optimizer,
    device,
):
    """
    Resume from latest checkpoint if available.

    Returns:

        resume_epoch
        best_metric
        best_epoch
    """

    if not os.path.exists(path):

        return (
            1,
            -1.0,
            0,
        )

    ckpt = torch.load(
        path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Restore model
    # --------------------------------------------------------

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    # --------------------------------------------------------
    # Restore optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        ckpt["optimizer_state_dict"]
    )

    # --------------------------------------------------------
    # Restore tracking information
    # --------------------------------------------------------

    resume_epoch = (
        ckpt["epoch"] + 1
    )

    best_metric = ckpt.get(
        "best_metric",
        -1.0,
    )

    best_epoch = ckpt.get(
        "best_epoch",
        0,
    )

    print(
        f"Found checkpoint at {path}"
    )

    print(
        f"Resuming from epoch {resume_epoch}"
    )

    print(
        "Checkpoint config: "
        f"{ckpt.get('dataset', 'unknown')}/"
        f"{ckpt.get('backbone', 'unknown')}/"
        f"{ckpt.get('loss_fn', 'unknown')}/"
        f"{ckpt.get('augmentation', 'unknown')}/"
        f"seed{ckpt.get('seed', 'unknown')}"
    )

    if ckpt.get("loss_fn") == "redl":

        print(
            f"Checkpoint R-EDL lambda: "
            f"{ckpt.get('redl_lambda', 'unknown')}"
        )

    print(
        f"Previous best validation accuracy: "
        f"{best_metric:.4f} "
        f"(epoch {best_epoch})"
    )

    return (
        resume_epoch,
        best_metric,
        best_epoch,
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    device,
    num_classes,
    loss_fn,
    redl_lambda=0.1,
):
    """
    Evaluate model using:

        - Accuracy
        - ECE
        - Reliability-diagram bin data
    """

    model.eval()

    all_conf = []
    all_pred = []
    all_label = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            output = model(
                images
            )

            # ------------------------------------------------
            # Standard EDL
            # ------------------------------------------------

            if loss_fn == "edl":

                pred_class, confidence, _ = (
                    edl_predictions(
                        output
                    )
                )

            # ------------------------------------------------
            # R-EDL
            # ------------------------------------------------

            elif loss_fn == "redl":

                pred_class, confidence, _ = (
                    redl_predictions(
                        output,
                        lam=redl_lambda,
                    )
                )

            # ------------------------------------------------
            # Softmax
            # ------------------------------------------------

            else:

                probs = torch.softmax(
                    output,
                    dim=1,
                )

                confidence, pred_class = (
                    torch.max(
                        probs,
                        dim=1,
                    )
                )

            all_conf.extend(
                confidence.cpu().numpy()
            )

            all_pred.extend(
                pred_class.cpu().numpy()
            )

            all_label.extend(
                labels.numpy()
            )

    all_pred = np.array(
        all_pred
    )

    all_label = np.array(
        all_label
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = float(
        np.mean(
            all_pred == all_label
        )
    )

    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    ece, bin_data = compute_ece(
        all_conf,
        all_pred,
        all_label,
        n_bins=15,
    )

    return (
        accuracy,
        ece,
        bin_data,
    )


# ============================================================
# CSV logging
# ============================================================

def write_log_row(
    log_path,
    log_row,
):
    """
    Append an experiment result to CSV.

    If an existing CSV has an incompatible header, a new file
    with '_v2' suffix is created instead of corrupting the
    previous results.
    """

    directory = os.path.dirname(
        log_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    fieldnames = list(
        log_row.keys()
    )

    actual_log_path = log_path

    # --------------------------------------------------------
    # Existing file
    # --------------------------------------------------------

    if os.path.exists(log_path):

        with open(
            log_path,
            "r",
            newline="",
        ) as f:

            reader = csv.reader(f)

            try:
                existing_header = next(
                    reader
                )
            except StopIteration:
                existing_header = []

        # ----------------------------------------------------
        # Compatible header
        # ----------------------------------------------------

        if existing_header == fieldnames:

            write_header = False

        # ----------------------------------------------------
        # Incompatible header
        # ----------------------------------------------------

        else:

            root, ext = os.path.splitext(
                log_path
            )

            actual_log_path = (
                f"{root}_v2{ext}"
            )

            print(
                f"Existing CSV schema differs."
            )

            print(
                f"Using new log file: "
                f"{actual_log_path}"
            )

            write_header = not os.path.exists(
                actual_log_path
            )

    else:

        write_header = True

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    with open(
        actual_log_path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if write_header:

            writer.writeheader()

        writer.writerow(
            log_row
        )

    return actual_log_path


# ============================================================
# Main training function
# ============================================================

def main():

    # ========================================================
    # Arguments
    # ========================================================

    args = parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Experiment information
    # ========================================================

    print(
        f"Experiment: "
        f"dataset={args.dataset} "
        f"backbone={args.backbone} "
        f"loss_fn={args.loss_fn} "
        f"augmentation={args.augmentation} "
        f"seed={args.seed}"
    )

    if args.loss_fn == "redl":

        print(
            f"R-EDL lambda: "
            f"{args.redl_lambda}"
        )

    # ========================================================
    # Directories
    # ========================================================

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(
            args.log_path
        ) or ".",
        exist_ok=True,
    )

    # ========================================================
    # Dataset check
    # ========================================================

    if args.dataset != "ham10000":

        raise NotImplementedError(
            "Only ham10000 is wired up right now. "
            "Add other datasets under datasets/."
        )

    # ========================================================
    # Number of classes
    # ========================================================

    num_classes = len(
        CLASS_NAMES
    )

    # ========================================================
    # Load training dataset
    # ========================================================

    full_dataset = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(
            train=True
        ),
    )

    # ========================================================
    # Load validation dataset
    # ========================================================

    val_dataset_raw = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(
            train=False
        ),
    )

    # ========================================================
    # Lesion-level split
    # ========================================================

    n = len(
        full_dataset
    )

    all_indices = np.arange(
        n
    )

    all_labels = np.array(
        full_dataset.labels
    )

    all_lesion_ids = (
        full_dataset.metadata[
            "lesion_id"
        ].values
    )

    # --------------------------------------------------------
    # Approximately 15% validation
    # --------------------------------------------------------

    n_splits = round(
        1 / args.val_split
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    train_indices, val_indices = next(
        sgkf.split(
            all_indices,
            all_labels,
            groups=all_lesion_ids,
        )
    )

    # ========================================================
    # Leakage check
    # ========================================================

    train_lesions = set(
        all_lesion_ids[
            train_indices
        ]
    )

    val_lesions = set(
        all_lesion_ids[
            val_indices
        ]
    )

    overlap = (
        train_lesions
        &
        val_lesions
    )

    print(
        f"Lesion overlap between train/val: "
        f"{len(overlap)} (should be 0)"
    )

    if len(overlap) != 0:

        raise RuntimeError(
            f"Lesion leakage detected: "
            f"{len(overlap)} lesion(s) appear "
            f"in both train and validation splits."
        )

    # ========================================================
    # Class distribution
    # ========================================================

    print(
        "Train class distribution:",
        pd.Series(
            all_labels[
                train_indices
            ]
        )
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    print(
        "Validation class distribution:",
        pd.Series(
            all_labels[
                val_indices
            ]
        )
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    # ========================================================
    # Debug subset
    # ========================================================

    if args.debug_subset is not None:

        if args.debug_subset <= 0:

            raise ValueError(
                "--debug_subset must be greater than 0."
            )

        if args.debug_subset > len(
            train_indices
        ):

            raise ValueError(
                "--debug_subset cannot exceed "
                "the number of training samples."
            )

        train_indices = (
            train_indices[
                :args.debug_subset
            ]
        )

    # ========================================================
    # Dataset subsets
    # ========================================================

    train_subset = Subset(
        full_dataset,
        train_indices,
    )

    val_subset = Subset(
        val_dataset_raw,
        val_indices,
    )

    # ========================================================
    # Class balancing
    # ========================================================

    train_labels = [
        full_dataset.labels[i]
        for i in train_indices
    ]

    class_counts = np.bincount(
        train_labels,
        minlength=num_classes,
    )

    class_weights = (
        1.0
        /
        np.maximum(
            class_counts,
            1,
        )
    )

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = (
        torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(
                sample_weights
            ),
            replacement=True,
        )
    )

    print(
        f"Train samples: "
        f"{len(train_subset)} "
        f"| Val samples: "
        f"{len(val_subset)} "
        f"| Sampler: "
        f"{type(sampler).__name__} "
        f"| Lesion-level split, "
        f"n_splits={n_splits}"
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ========================================================
    # Model
    # ========================================================

    model = get_backbone(
        args.backbone,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # ========================================================
    # Checkpoint paths
    # ========================================================

    ckpt_path = checkpoint_path(
        args.checkpoint_dir,
        args.dataset,
        args.backbone,
        args.loss_fn,
        args.augmentation,
        args.seed,
        args.debug_subset,
        args.redl_lambda,
    )

    best_path = best_checkpoint_path(
        ckpt_path
    )

    print(
        f"Checkpoint path: "
        f"{ckpt_path}"
    )

    print(
        f"Best checkpoint path: "
        f"{best_path}"
    )

    # ========================================================
    # Resume
    # ========================================================

    (
        start_epoch,
        best_val_acc,
        best_epoch,
    ) = load_checkpoint_if_exists(
        ckpt_path,
        model,
        optimizer,
        device,
    )

    # ========================================================
    # Early stopping
    # ========================================================

    epochs_no_improve = 0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        model.train()

        epoch_start = time.time()

        running_loss = 0.0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            output = model(
                images
            )

            # =================================================
            # Standard EDL
            # =================================================

            if args.loss_fn == "edl":

                loss = edl_mse_loss(
                    output,
                    labels,
                    epoch_num=epoch,
                    num_classes=num_classes,
                    annealing_step=args.annealing_step,
                    device=device,
                )

            # =================================================
            # R-EDL
            # =================================================

            elif args.loss_fn == "redl":

                loss = redl_loss(
                    output,
                    labels,
                    epoch_num=epoch,
                    num_classes=num_classes,
                    annealing_step=args.annealing_step,
                    device=device,
                    lam=args.redl_lambda,
                )

            # =================================================
            # Softmax
            # =================================================

            else:

                loss = nn.functional.cross_entropy(
                    output,
                    labels,
                )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                *
                images.size(0)
            )

        # ====================================================
        # Average training loss
        # ====================================================

        train_loss = (
            running_loss
            /
            len(train_subset)
        )

        # ====================================================
        # Validation
        # ====================================================

        val_acc, val_ece, _ = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            args.loss_fn,
            redl_lambda=args.redl_lambda,
        )

        elapsed = (
            time.time()
            -
            epoch_start
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_ece={val_ece:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ====================================================
        # Best-model detection
        # ====================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch

            epochs_no_improve = 0

            print(
                f"  -> New best validation accuracy: "
                f"{best_val_acc:.4f} "
                f"(epoch {best_epoch})"
            )

            # ------------------------------------------------
            # Save BEST checkpoint
            # ------------------------------------------------

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                args,
                best_val_acc,
                best_epoch,
            )

            print(
                f"  best checkpoint saved -> "
                f"{best_path}"
            )

        else:

            epochs_no_improve += 1

        # ====================================================
        # Save latest/recovery checkpoint
        # ====================================================

        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            args,
            best_val_acc,
            best_epoch,
        )

        print(
            f"  checkpoint saved -> "
            f"{ckpt_path}"
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if epochs_no_improve >= args.patience:

            print(
                f"Early stopping: "
                f"no validation improvement "
                f"in {args.patience} epochs."
            )

            break

    # ========================================================
    # Reload BEST checkpoint
    # ========================================================

    if os.path.exists(
        best_path
    ):

        print()
        print(
            f"Loading best checkpoint "
            f"for final evaluation:"
        )
        print(
            best_path
        )

        best_ckpt = torch.load(
            best_path,
            map_location=device,
        )

        model.load_state_dict(
            best_ckpt[
                "model_state_dict"
            ]
        )

        print(
            f"Best checkpoint epoch: "
            f"{best_ckpt.get('epoch', 'unknown')}"
        )

    else:

        print(
            "WARNING: Best checkpoint not found. "
            "Using current model for final evaluation."
        )

    # ========================================================
    # Final evaluation
    # ========================================================

    final_acc, final_ece, _ = evaluate(
        model,
        val_loader,
        device,
        num_classes,
        args.loss_fn,
        redl_lambda=args.redl_lambda,
    )

    # ========================================================
    # Run ID
    # ========================================================

    run_id = (
        f"{args.dataset}_"
        f"{args.backbone}_"
        f"{args.loss_fn}_"
        f"{args.augmentation}_"
        f"seed{args.seed}"
    )

    if args.loss_fn == "redl":

        run_id += (
            f"_lam{args.redl_lambda:g}"
        )

    # ========================================================
    # CSV log row
    # ========================================================

    log_row = {

        "run_id":
            run_id,

        "dataset":
            args.dataset,

        "backbone":
            args.backbone,

        "loss_fn":
            args.loss_fn,

        "augmentation":
            args.augmentation,

        "seed":
            args.seed,

        "epochs":
            epoch,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.lr,

        "annealing_step":
            args.annealing_step,

        "redl_lambda":
            args.redl_lambda,

        "train_samples":
            len(train_subset),

        "val_samples":
            len(val_subset),

        "accuracy":
            final_acc,

        "ece":
            final_ece,

        "ood_auroc":
            "",

        "latency":
            "",

        "best_epoch":
            best_epoch,

        "best_val_accuracy":
            best_val_acc,

        "checkpoint_path":
            ckpt_path,

        "best_checkpoint_path":
            best_path,
    }

    # ========================================================
    # Write CSV
    # ========================================================

    actual_log_path = write_log_row(
        args.log_path,
        log_row,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Loss function            : "
        f"{args.loss_fn}"
    )

    if args.loss_fn == "redl":

        print(
            f"R-EDL lambda             : "
            f"{args.redl_lambda}"
        )

    print(
        f"Best validation accuracy : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Best epoch               : "
        f"{best_epoch}"
    )

    print(
        f"Final validation accuracy: "
        f"{final_acc:.4f}"
    )

    print(
        f"Final validation ECE     : "
        f"{final_ece:.4f}"
    )

    print(
        f"Results logged to        : "
        f"{actual_log_path}"
    )

    print(
        f"Latest checkpoint        : "
        f"{ckpt_path}"
    )

    print(
        f"Best checkpoint          : "
        f"{best_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()