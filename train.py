"""
Train an EDL-head or softmax backbone on HAM10000.

Checkpoint strategy
-------------------
Every experiment permutation gets its own checkpoint identity:

    {dataset}_{backbone}_{loss_fn}_{augmentation}_seed{seed}.pt

Two checkpoint files are maintained:

1. Latest/recovery checkpoint:
       ..._seed42.pt

   Overwritten every epoch.
   Used to resume training after a Colab disconnect.

2. Best checkpoint:
       ..._seed42_best.pt

   Saved only when validation accuracy improves.
   Always contains the actual best-performing model weights.

The full experiment configuration is stored inside both checkpoint files.

Example:

    python train.py \
        --backbone efficientnet_b0 \
        --loss_fn edl \
        --dataset ham10000 \
        --augmentation standard \
        --seed 42 \
        --data_root data/ham10000 \
        --epochs 20 \
        --checkpoint_dir /content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints \
        --log_path /content/drive/MyDrive/Lightweight-EDL-Medical/results/master_log.csv
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets.ham10000 import (
    HAM10000Dataset,
    default_transforms,
    CLASS_NAMES,
)

from losses.evidential_loss import (
    edl_mse_loss,
    edl_predictions,
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
        description="Train EDL or Softmax model on HAM10000."
    )

    p.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=[
            "efficientnet_b0",
            "mobilenet_v3_small",
            "shufflenet_v2",
        ],
    )

    p.add_argument(
        "--loss_fn",
        default="edl",
        choices=["edl", "softmax"],
    )

    p.add_argument(
        "--dataset",
        default="ham10000",
    )

    p.add_argument(
        "--augmentation",
        default="standard",
        help=(
            "Label for the augmentation strategy used "
            "(e.g. standard, mixup, cutmix, randaugment). "
            "Actual augmentation logic should be implemented "
            "in datasets/*.py."
        ),
    )

    p.add_argument(
        "--data_root",
        default="data/ham10000",
    )

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

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--val_split",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--debug_subset",
        type=int,
        default=None,
        help=(
            "If set, train using only N samples "
            "for a fast sanity check."
        ),
    )

    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )

    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help=(
            "Directory where experiment checkpoints are stored. "
            "Use a Google Drive path in Colab."
        ),
    )

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
):
    """
    Generate a unique checkpoint filename for each experiment.

    Example:
        ham10000_efficientnet_b0_edl_standard_seed42.pt

    Debug/sanity-check runs get a distinguishing suffix so they
    never collide with a real full-dataset run of the same config:
        ham10000_efficientnet_b0_edl_standard_seed42_debug200.pt
    """

    debug_suffix = f"_debug{debug_subset}" if debug_subset else ""

    fname = (
        f"{dataset}_"
        f"{backbone}_"
        f"{loss_fn}_"
        f"{augmentation}_"
        f"seed{seed}"
        f"{debug_suffix}.pt"
    )

    return os.path.join(
        checkpoint_dir,
        fname,
    )


def best_checkpoint_path(ckpt_path):
    """
    Given the regular latest/recovery checkpoint path,
    return the matching '_best' checkpoint path.

    Example:

        ham10000_efficientnet_b0_edl_standard_seed42.pt

    becomes:

        ham10000_efficientnet_b0_edl_standard_seed42_best.pt

    The regular checkpoint contains the latest epoch.

    The best checkpoint contains the actual model weights
    from the epoch with the best validation accuracy.
    """

    root, ext = os.path.splitext(ckpt_path)

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
    Save model weights, optimizer state, experiment configuration,
    and best-validation-result tracking.
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

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            # ------------------------------------------------
            # Experiment identity
            # ------------------------------------------------
            "dataset": args.dataset,

            "backbone": args.backbone,

            "loss_fn": args.loss_fn,

            "augmentation": args.augmentation,

            "seed": args.seed,

            # ------------------------------------------------
            # Hyperparameters
            # ------------------------------------------------
            "batch_size": args.batch_size,

            "lr": args.lr,

            "annealing_step": args.annealing_step,

            "val_split": args.val_split,

            # ------------------------------------------------
            # Best validation result
            # ------------------------------------------------
            "best_metric": best_metric,

            "best_epoch": best_epoch,
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
    Resume training if a latest checkpoint exists.

    Returns:

        resume_epoch
        best_metric
        best_epoch

    If no checkpoint exists:

        1
        -1.0
        0

    The best validation information is restored from the
    checkpoint so a Colab restart does not forget the previous
    best epoch.
    """

    if not os.path.exists(path):

        return 1, -1.0, 0

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
    # Restore training information
    # --------------------------------------------------------

    resume_epoch = ckpt["epoch"] + 1

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
):
    """
    Evaluate the model using:

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

            images = images.to(device)

            output = model(images)

            # ------------------------------------------------
            # EDL prediction
            # ------------------------------------------------

            if loss_fn == "edl":

                pred_class, confidence, _ = edl_predictions(
                    output
                )

            # ------------------------------------------------
            # Softmax prediction
            # ------------------------------------------------

            else:

                probs = torch.softmax(
                    output,
                    dim=1,
                )

                confidence, pred_class = torch.max(
                    probs,
                    dim=1,
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
# Main training function
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        f"Experiment: "
        f"dataset={args.dataset} "
        f"backbone={args.backbone} "
        f"loss_fn={args.loss_fn} "
        f"augmentation={args.augmentation} "
        f"seed={args.seed}"
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(args.log_path)
        or ".",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset check
    # --------------------------------------------------------

    if args.dataset != "ham10000":

        raise NotImplementedError(
            "Only ham10000 is wired up right now. "
            "Add other datasets under datasets/."
        )

    # --------------------------------------------------------
    # Number of classes
    # --------------------------------------------------------

    num_classes = len(
        CLASS_NAMES
    )

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    full_dataset = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(
            train=True
        ),
    )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    val_dataset_raw = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(
            train=False
        ),
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    n = len(
        full_dataset
    )

    indices = list(
        range(n)
    )

    random.shuffle(
        indices
    )

    # --------------------------------------------------------
    # Debug subset
    # --------------------------------------------------------

    if args.debug_subset is not None:

        if args.debug_subset <= 0:

            raise ValueError(
                "--debug_subset must be greater than 0."
            )

        indices = indices[
            :args.debug_subset
        ]

        n = len(
            indices
        )

    # --------------------------------------------------------
    # Validation size
    # --------------------------------------------------------

    n_val = max(
        1,
        int(
            n * args.val_split
        ),
    )

    if n_val >= n:

        raise ValueError(
            "Validation split leaves no training samples. "
            "Increase dataset size or decrease --val_split."
        )

    # --------------------------------------------------------
    # Indices
    # --------------------------------------------------------

    val_indices = indices[
        :n_val
    ]

    train_indices = indices[
        n_val:
    ]

    # --------------------------------------------------------
    # Dataset subsets
    # --------------------------------------------------------

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
        / np.maximum(
            class_counts,
            1,
        )
    )

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    print(
        f"Train samples: {len(train_subset)} "
        f"| Val samples: {len(val_subset)} "
        f"| Sampler: {type(sampler).__name__}"
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
    )

    best_path = best_checkpoint_path(
        ckpt_path
    )

    print(
        f"Checkpoint path: {ckpt_path}"
    )

    print(
        f"Best checkpoint path: {best_path}"
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

    # --------------------------------------------------------
    # Early stopping counter
    # --------------------------------------------------------

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

            # ------------------------------------------------
            # EDL loss
            # ------------------------------------------------

            if args.loss_fn == "edl":

                loss = edl_mse_loss(
                    output,
                    labels,
                    epoch_num=epoch,
                    num_classes=num_classes,
                    annealing_step=args.annealing_step,
                    device=device,
                )

            # ------------------------------------------------
            # Softmax loss
            # ------------------------------------------------

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
                * images.size(0)
            )

        # ----------------------------------------------------
        # Average training loss
        # ----------------------------------------------------

        train_loss = (
            running_loss
            / len(train_subset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_acc, val_ece, _ = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            args.loss_fn,
        )

        elapsed = (
            time.time()
            - epoch_start
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
            # Save actual BEST model
            #
            # This checkpoint is NOT overwritten on epochs
            # where validation accuracy does not improve.
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
        # Save LATEST / RECOVERY checkpoint
        #
        # This is intentionally overwritten every epoch for
        # THIS experiment permutation only.
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
    # Final evaluation
    # ========================================================

    final_acc, final_ece, _ = evaluate(
        model,
        val_loader,
        device,
        num_classes,
        args.loss_fn,
    )

    # ========================================================
    # CSV logging
    # ========================================================

    run_id = (
        f"{args.dataset}_"
        f"{args.backbone}_"
        f"{args.loss_fn}_"
        f"{args.augmentation}_"
        f"seed{args.seed}"
    )

    log_row = {

        "run_id": run_id,

        "dataset": args.dataset,

        "backbone": args.backbone,

        "loss_fn": args.loss_fn,

        "augmentation": args.augmentation,

        "seed": args.seed,

        "epochs": epoch,

        "batch_size": args.batch_size,

        "learning_rate": args.lr,

        "annealing_step": args.annealing_step,

        "train_samples": len(
            train_subset
        ),

        "val_samples": len(
            val_subset
        ),

        "accuracy": final_acc,

        "ece": final_ece,

        "ood_auroc": "",

        "latency": "",

        "best_epoch": best_epoch,

        "best_val_accuracy": best_val_acc,

        "checkpoint_path": ckpt_path,

        "best_checkpoint_path": best_path,
    }

    # ========================================================
    # Write CSV
    # ========================================================

    write_header = not os.path.exists(
        args.log_path
    )

    with open(
        args.log_path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                log_row.keys()
            ),
        )

        if write_header:

            writer.writeheader()

        writer.writerow(
            log_row
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
        f"{args.log_path}"
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