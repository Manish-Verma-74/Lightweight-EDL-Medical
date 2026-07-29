"""
Train an EDL-head backbone on HAM10000.

Usage:
    PYTHONPATH=. python train.py --backbone efficientnet_b0 --dataset ham10000 \
        --data_root data/ham10000 --epochs 2 --debug_subset 200

    PYTHONPATH=. python train.py --backbone efficientnet_b0 --dataset ham10000 \
        --data_root data/ham10000 --epochs 50
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

from datasets.ham10000 import HAM10000Dataset, default_transforms, CLASS_NAMES
from losses.evidential_loss import edl_mse_loss, edl_predictions
from metrics.ece import compute_ece
from models.backbone_factory import get_backbone


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    p.add_argument("--dataset", default="ham10000")
    p.add_argument("--data_root", default="data/ham10000")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--annealing_step", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--debug_subset", type=int, default=None, help="If set, train on only N samples (fast sanity check)")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs with no val improvement)")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_path", default="results/master_log.csv")
    return p.parse_args()


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_conf, all_pred, all_label = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            pred_class, confidence, _ = edl_predictions(output)
            all_conf.extend(confidence.cpu().numpy())
            all_pred.extend(pred_class.cpu().numpy())
            all_label.extend(labels.numpy())

    all_pred = np.array(all_pred)
    all_label = np.array(all_label)
    accuracy = float(np.mean(all_pred == all_label))
    ece, bin_data = compute_ece(all_conf, all_pred, all_label, n_bins=15)
    return accuracy, ece, bin_data


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # --- Data ---
    if args.dataset != "ham10000":
        raise NotImplementedError("Only ham10000 is wired up right now -- add other datasets under datasets/")

    num_classes = len(CLASS_NAMES)
    full_dataset = HAM10000Dataset(args.data_root, transform=default_transforms(train=True))
    val_dataset_raw = HAM10000Dataset(args.data_root, transform=default_transforms(train=False))

    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)

    if args.debug_subset:
        indices = indices[: args.debug_subset]
        n = len(indices)

    n_val = max(1, int(n * args.val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(val_dataset_raw, val_indices)

    # Weighted sampling for class imbalance, restricted to the train subset
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")

    # --- Model ---
    model = get_backbone(args.backbone, num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(args.checkpoint_dir, f"{args.backbone}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = edl_mse_loss(output, labels, epoch_num=epoch, num_classes=num_classes,
                                 annealing_step=args.annealing_step, device=device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_subset)
        val_acc, val_ece, _ = evaluate(model, val_loader, device, num_classes)
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_ece={val_ece:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> new best (val_acc={val_acc:.4f}), checkpoint saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no val improvement in {args.patience} epochs.")
                break

    # --- Final logging ---
    final_acc, final_ece, _ = evaluate(model, val_loader, device, num_classes)
    log_row = {
        "backbone": args.backbone,
        "dataset": args.dataset,
        "epochs_run": epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_acc,
        "final_val_ece": final_ece,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "debug_subset": args.debug_subset,
        "checkpoint": best_ckpt_path,
    }

    write_header = not os.path.exists(args.log_path)
    with open(args.log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    print(f"\nDone. Best val_acc={best_val_acc:.4f}. Logged to {args.log_path}")


if __name__ == "__main__":
    main()
