"""
Train an EDL, R-EDL, F-EDL, or Softmax lightweight backbone on HAM10000.

Google Drive Persistent Storage Architecture:
    /content/drive/MyDrive/Lightweight-EDL-Medical/
    ├── checkpoints/
    ├── results/
    │   └── master_log.csv
    └── data/
        └── ham10000/

Supported Loss Functions:
    - edl      : Standard Evidential Deep Learning (Sensoy et al., NeurIPS 2018)
    - redl     : Relaxed Evidential Deep Learning (Chen et al., ICLR 2024)
    - fedl     : Flexible Evidential Deep Learning (Yoon & Kim, NeurIPS 2025)
    - softmax  : Standard Cross-Entropy baseline

Supported Lightweight Backbones:
    - efficientnet_b0
    - mobilenet_v3_small
    - shufflenet_v2

Example Colab Execution (F-EDL Screening on EfficientNet-B0):

    python train.py \
        --backbone efficientnet_b0 \
        --loss_fn fedl \
        --dataset ham10000 \
        --augmentation standard \
        --seed 42 \
        --data_root /content/ham10000 \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --patience 5 \
        --checkpoint_dir /content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints \
        --log_path /content/drive/MyDrive/Lightweight-EDL-Medical/results/master_log.csv
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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

from losses.fedl_loss import (
    fedl_loss,
    fedl_predictions,
)

from metrics.ece import compute_ece
from models.backbone_factory import get_backbone
from models.fedl_wrapper import FEDLWrapper


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
# Command-line Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train EDL, R-EDL, F-EDL, or Softmax model on HAM10000 in Google Colab."
    )

    p.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"],
    )

    p.add_argument(
        "--loss_fn",
        default="edl",
        choices=["edl", "redl", "softmax", "fedl"],
    )

    p.add_argument("--dataset", default="ham10000")

    p.add_argument(
        "--augmentation",
        default="standard",
        help="Label for the augmentation strategy (e.g. standard, mixup, cutmix, randaugment).",
    )

    p.add_argument(
        "--data_root",
        default="/content/ham10000",
        help="Root directory containing HAM10000 images and metadata.",
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--annealing_step", type=int, default=10)

    p.add_argument(
        "--redl_lambda",
        type=float,
        default=0.1,
        help="Prior weight hyperparameter for R-EDL. Used only when --loss_fn redl.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.15)

    p.add_argument(
        "--debug_subset",
        type=int,
        default=None,
        help="If set, train using only N samples for a fast sanity check.",
    )

    p.add_argument("--patience", type=int, default=5, help="Early stopping patience.")

    p.add_argument(
        "--checkpoint_dir",
        default="/content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints",
        help="Google Drive directory where experiment checkpoints are saved.",
    )

    p.add_argument(
        "--log_path",
        default="/content/drive/MyDrive/Lightweight-EDL-Medical/results/master_log.csv",
        help="Google Drive path to the master CSV experiment log.",
    )

    return p.parse_args()


# ============================================================
# Checkpoint Naming Logic
# ============================================================

def checkpoint_path(
    checkpoint_dir, dataset, backbone, loss_fn, augmentation, seed, debug_subset=None, redl_lambda=0.1,
):
    debug_suffix = f"_debug{debug_subset}" if debug_subset is not None else ""
    lambda_suffix = f"_lam{redl_lambda:g}" if loss_fn == "redl" else ""
    fname = f"{dataset}_{backbone}_{loss_fn}_{augmentation}_seed{seed}{lambda_suffix}{debug_suffix}.pt"
    return os.path.join(checkpoint_dir, fname)


def best_checkpoint_path(ckpt_path):
    root, ext = os.path.splitext(ckpt_path)
    return f"{root}_best{ext}"


# ============================================================
# Checkpoint Save & Load
# ============================================================

def save_checkpoint(path, model, optimizer, epoch, args, best_metric, best_epoch):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataset": args.dataset,
            "backbone": args.backbone,
            "loss_fn": args.loss_fn,
            "augmentation": args.augmentation,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "annealing_step": args.annealing_step,
            "redl_lambda": args.redl_lambda,
            "val_split": args.val_split,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        },
        path,
    )


def load_checkpoint_if_exists(path, model, optimizer, device):
    if not os.path.exists(path):
        return 1, -1.0, 0

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    resume_epoch = ckpt["epoch"] + 1
    best_metric = ckpt.get("best_metric", -1.0)
    best_epoch = ckpt.get("best_epoch", 0)

    print(f"Found existing checkpoint at: {path}")
    print(f"Resuming execution from epoch {resume_epoch}")
    print(
        "Checkpoint Config: "
        f"{ckpt.get('dataset', 'unknown')}/"
        f"{ckpt.get('backbone', 'unknown')}/"
        f"{ckpt.get('loss_fn', 'unknown')}/"
        f"{ckpt.get('augmentation', 'unknown')}/"
        f"seed{ckpt.get('seed', 'unknown')}"
    )
    if ckpt.get("loss_fn") == "redl":
        print(f"Checkpoint R-EDL lambda: {ckpt.get('redl_lambda', 'unknown')}")
    if ckpt.get("loss_fn") == "fedl":
        print("Checkpoint F-EDL configuration detected.")

    print(f"Previous best validation accuracy: {best_metric:.4f} (epoch {best_epoch})")
    return resume_epoch, best_metric, best_epoch


# ============================================================
# Model Evaluation
# ============================================================

def evaluate(model, loader, device, num_classes, loss_fn, redl_lambda=0.1):
    model.eval()
    all_conf, all_pred, all_label = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)

            if loss_fn == "edl":
                pred_class, confidence, _ = edl_predictions(output)
            elif loss_fn == "redl":
                pred_class, confidence, _ = redl_predictions(output, lam=redl_lambda)
            elif loss_fn == "fedl":
                alpha, p, tau = output
                pred_class, confidence, *_ = fedl_predictions(alpha, p, tau)
            elif loss_fn == "softmax":
                probs = torch.softmax(output, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")

            all_conf.extend(confidence.detach().cpu().numpy())
            all_pred.extend(pred_class.detach().cpu().numpy())
            all_label.extend(labels.cpu().numpy())

    all_pred = np.array(all_pred)
    all_label = np.array(all_label)

    accuracy = float(np.mean(all_pred == all_label))
    ece, bin_data = compute_ece(all_conf, all_pred, all_label, n_bins=15)

    return accuracy, ece, bin_data


# ============================================================
# Master CSV Writer
# ============================================================

def write_log_row(log_path, log_row):
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = list(log_row.keys())
    actual_log_path = log_path

    if os.path.exists(log_path):
        with open(log_path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

        if existing_header == fieldnames:
            write_header = False
        else:
            root, ext = os.path.splitext(log_path)
            actual_log_path = f"{root}_v2{ext}"
            print(f"CSV header mismatch detected. Diverting entry to: {actual_log_path}")
            write_header = not os.path.exists(actual_log_path)
    else:
        write_header = True

    with open(actual_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    return actual_log_path


# ============================================================
# Main Loop
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    print(
        f"Experiment setup: dataset={args.dataset} | backbone={args.backbone} | "
        f"loss_fn={args.loss_fn} | augmentation={args.augmentation} | seed={args.seed}"
    )

    if args.loss_fn == "redl":
        print(f"R-EDL lambda hyperparameter: {args.redl_lambda}")
    if args.loss_fn == "fedl":
        print("F-EDL mode active: controlled adaptation (no spectral normalization).")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    if args.dataset != "ham10000":
        raise NotImplementedError("Only ham10000 is supported in this pipeline.")

    num_classes = len(CLASS_NAMES)

    full_dataset = HAM10000Dataset(args.data_root, transform=default_transforms(train=True))
    val_dataset_raw = HAM10000Dataset(args.data_root, transform=default_transforms(train=False))

    n = len(full_dataset)
    all_indices = np.arange(n)
    all_labels = np.array(full_dataset.labels)
    all_lesion_ids = full_dataset.metadata["lesion_id"].values

    # Stratified Group K-Fold split (zero lesion leakage)
    n_splits = round(1 / args.val_split)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    train_indices, val_indices = next(sgkf.split(all_indices, all_labels, groups=all_lesion_ids))

    train_lesions = set(all_lesion_ids[train_indices])
    val_lesions = set(all_lesion_ids[val_indices])
    overlap = train_lesions & val_lesions
    print(f"Lesion overlap between train and validation splits: {len(overlap)} (must be 0)")
    if len(overlap) != 0:
        raise RuntimeError(f"Lesion leakage detected! {len(overlap)} lesion(s) appear in both splits.")

    print("Train class distribution:", pd.Series(all_labels[train_indices]).value_counts().sort_index().to_dict())
    print("Val class distribution:  ", pd.Series(all_labels[val_indices]).value_counts().sort_index().to_dict())

    if args.debug_subset is not None:
        if args.debug_subset <= 0 or args.debug_subset > len(train_indices):
            raise ValueError(f"--debug_subset must be between 1 and {len(train_indices)}.")
        train_indices = train_indices[:args.debug_subset]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset_raw, val_indices)

    # Class balancing via WeightedRandomSampler
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    print(
        f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)} "
        f"| Sampler: {type(sampler).__name__} | Lesion-level split (n_splits={n_splits})"
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model instantiation
    if args.loss_fn == "fedl":
        model = FEDLWrapper(backbone_name=args.backbone, num_classes=num_classes, pretrained=True).to(device)
        print("F-EDL mode active: using FEDLWrapper (alpha, p, tau outputs).")
    else:
        model = get_backbone(args.backbone, num_classes=num_classes, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_path = checkpoint_path(
        args.checkpoint_dir, args.dataset, args.backbone, args.loss_fn,
        args.augmentation, args.seed, args.debug_subset, args.redl_lambda,
    )
    best_path = best_checkpoint_path(ckpt_path)

    print(f"Latest checkpoint target: {ckpt_path}")
    print(f"Best checkpoint target:   {best_path}")

    start_epoch, best_val_acc, best_epoch = load_checkpoint_if_exists(ckpt_path, model, optimizer, device)
    epochs_no_improve = 0
    last_epoch = start_epoch - 1

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)

            if args.loss_fn == "edl":
                loss = edl_mse_loss(
                    output, labels, epoch_num=epoch, num_classes=num_classes,
                    annealing_step=args.annealing_step, device=device,
                )
            elif args.loss_fn == "redl":
                loss = redl_loss(
                    output, labels, epoch_num=epoch, num_classes=num_classes,
                    annealing_step=args.annealing_step, device=device, lam=args.redl_lambda,
                )
            elif args.loss_fn == "fedl":
                alpha, p, tau = output
                loss = fedl_loss(alpha, p, tau, labels, num_classes)
            elif args.loss_fn == "softmax":
                loss = nn.functional.cross_entropy(output, labels)
            else:
                raise ValueError(f"Unsupported loss function: {args.loss_fn}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_subset)
        val_acc, val_ece, _ = evaluate(
            model, val_loader, device, num_classes, args.loss_fn, redl_lambda=args.redl_lambda,
        )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_ece={val_ece:.4f} | {elapsed:.1f}s"
        )

        last_epoch = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"  -> New best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            save_checkpoint(best_path, model, optimizer, epoch, args, best_val_acc, best_epoch)
            print(f"  Best checkpoint saved -> {best_path}")
        else:
            epochs_no_improve += 1

        save_checkpoint(ckpt_path, model, optimizer, epoch, args, best_val_acc, best_epoch)
        print(f"  Latest checkpoint saved -> {ckpt_path}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered: no validation improvement in {args.patience} epochs.")
            break

    # Reload Best Model for Final Logging
    if os.path.exists(best_path):
        print(f"\nReloading best checkpoint for final evaluation:\n{best_path}")
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Best checkpoint epoch: {best_ckpt.get('epoch', 'unknown')}")
    else:
        print("\nWARNING: Best checkpoint file not found. Evaluating current state.")

    final_acc, final_ece, _ = evaluate(
        model, val_loader, device, num_classes, args.loss_fn, redl_lambda=args.redl_lambda,
    )

    run_id = f"{args.dataset}_{args.backbone}_{args.loss_fn}_{args.augmentation}_seed{args.seed}"
    if args.loss_fn == "redl":
        run_id += f"_lam{args.redl_lambda:g}"

    log_row = {
        "run_id": run_id,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "loss_fn": args.loss_fn,
        "augmentation": args.augmentation,
        "seed": args.seed,
        "epochs": last_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "annealing_step": args.annealing_step,
        "redl_lambda": args.redl_lambda,
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "accuracy": final_acc,
        "ece": final_ece,
        "ood_auroc": "",
        "latency": "",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "checkpoint_path": ckpt_path,
        "best_checkpoint_path": best_path,
    }

    actual_log_path = write_log_row(args.log_path, log_row)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Loss function            : {args.loss_fn}")
    if args.loss_fn == "redl":
        print(f"R-EDL lambda             : {args.redl_lambda}")
    if args.loss_fn == "fedl":
        print("F-EDL adaptation         : controlled, no spectral normalization")
    print(f"Best validation accuracy : {best_val_acc:.4f}")
    print(f"Best epoch               : {best_epoch}")
    print(f"Final validation accuracy: {final_acc:.4f}")
    print(f"Final validation ECE     : {final_ece:.4f}")
    print(f"Results logged to        : {actual_log_path}")
    print(f"Latest checkpoint        : {ckpt_path}")
    print(f"Best checkpoint          : {best_path}")


if __name__ == "__main__":
    main()