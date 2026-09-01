"""
Check calibration of the trained HAM10000 models using the
LESION-LEVEL corrected split (Task #1 EDL, Task #2 softmax).

Loads:
    1. EfficientNet-B0 + EDL checkpoint   (..._edl_standard_lesion_seed42_best.pt)
    2. EfficientNet-B0 + Softmax checkpoint (..._softmax_standard_lesion_seed42_best.pt)

Evaluates on the SAME lesion-grouped validation split used during training
(same seed, same StratifiedGroupKFold config), so results are directly
comparable between models and consistent with what train.py logged.

This script DOES NOT train the models again.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from datasets.ham10000 import HAM10000Dataset, default_transforms
from losses.evidential_loss import edl_predictions
from metrics.ece import compute_ece, plot_reliability_diagram
from models.backbone_factory import get_backbone
from train import checkpoint_path, best_checkpoint_path

HAM10000_NUM_CLASSES = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    p.add_argument("--dataset", default="ham10000")
    p.add_argument("--augmentation", default="standard_lesion")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--checkpoint_dir",
                    default="/content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints")
    p.add_argument("--data_root", default="data/ham10000")
    p.add_argument("--output_dir", default="results/reliability_diagrams")
    return p.parse_args()


def build_val_loader(args):
    """Rebuild the exact same lesion-grouped validation split train.py used."""
    val_dataset_raw = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(train=False),
    )

    n = len(val_dataset_raw)
    all_indices = np.arange(n)
    all_labels = np.array(val_dataset_raw.labels)
    all_lesion_ids = val_dataset_raw.metadata["lesion_id"].values

    n_splits = round(1 / args.val_split)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    _, val_indices = next(sgkf.split(all_indices, all_labels, groups=all_lesion_ids))

    val_subset = Subset(val_dataset_raw, val_indices)
    print(f"Rebuilt lesion-level validation split: {len(val_subset)} samples "
          f"(seed={args.seed}, n_splits={n_splits})")

    return DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)


def load_trained_model(ckpt_path, backbone, num_classes, device):
    model = get_backbone(backbone, num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from {ckpt_path} (trained through epoch {ckpt['epoch']}, "
          f"best_metric={ckpt.get('best_metric')})")
    model.eval()
    return model


def collect_predictions(model, loader, device, loss_fn):
    all_conf, all_pred, all_label = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            if loss_fn == "edl":
                pred_class, confidence, _ = edl_predictions(output)
            else:
                probs = torch.softmax(output, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
            all_conf.extend(confidence.cpu().numpy())
            all_pred.extend(pred_class.cpu().numpy())
            all_label.extend(labels.numpy())
    return np.array(all_conf), np.array(all_pred), np.array(all_label)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_loader = build_val_loader(args)

    results_summary = []

    for loss_fn in ["edl", "softmax"]:
        print(f"\n=== Reliability diagram: {args.backbone} ({loss_fn}, lesion-level) ===")

        ckpt_path = checkpoint_path(
            args.checkpoint_dir, args.dataset, args.backbone,
            loss_fn, args.augmentation, args.seed,
        )
        best_path = best_checkpoint_path(ckpt_path)

        model = load_trained_model(best_path, args.backbone, HAM10000_NUM_CLASSES, device)

        confidences, predictions, labels = collect_predictions(model, val_loader, device, loss_fn)
        accuracy = float(np.mean(predictions == labels))
        avg_confidence = float(np.mean(confidences))
        ece, bin_data = compute_ece(confidences, predictions, labels, n_bins=args.n_bins)

        print(f"  accuracy={accuracy:.4f} | avg_confidence={avg_confidence:.4f} | ECE={ece:.4f}")
        direction = "UNDERconfident" if avg_confidence < accuracy else "OVERconfident"
        print(f"  Direction of miscalibration: {direction} "
              f"(avg confidence {avg_confidence:.4f} vs accuracy {accuracy:.4f})")

        results_summary.append((loss_fn, accuracy, avg_confidence, ece, direction))

        save_path = os.path.join(
            args.output_dir, f"ham10000_{args.backbone}_{loss_fn}_lesion_reliability.png"
        )
        plot_reliability_diagram(
            bin_data, n_bins=args.n_bins, save_path=save_path,
            title=f"HAM10000 (lesion-level) {args.backbone} -- {loss_fn} "
                  f"(ECE={ece:.4f}, {direction})"
        )
        print(f"  Saved reliability diagram -> {save_path}")

    print("\n=== Summary (best-checkpoint, lesion-level split, consistent evaluation) ===")
    for loss_fn, acc, conf, ece, direction in results_summary:
        print(f"  {loss_fn:8s} | accuracy={acc:.4f} | avg_confidence={conf:.4f} "
              f"| ECE={ece:.4f} | {direction}")

    print("\nDone. Compare the two saved PNGs.")


if __name__ == "__main__":
    main()