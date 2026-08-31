"""
Check calibration of the trained HAM10000 models (Task #1 EDL, Task #2 softmax).

Loads:
    1. EfficientNet-B0 + EDL checkpoint (ham10000_efficientnet_b0_edl_standard_seed42_best.pt)
    2. EfficientNet-B0 + Softmax checkpoint (ham10000_efficientnet_b0_softmax_standard_seed42_best.pt)

Evaluates on the SAME validation split used during training (same seed, same
val_split ratio) so the reported ECE matches what was logged in master_log.csv.

This script DOES NOT train the models again.
"""

import argparse
import os
import random

import numpy as np
import torch
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
    p.add_argument("--augmentation", default="standard")
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
    """Rebuild the exact same validation split train.py used, via the same seed."""
    random.seed(args.seed)

    val_dataset_raw = HAM10000Dataset(
        args.data_root,
        transform=default_transforms(train=False),
    )

    n = len(val_dataset_raw)
    indices = list(range(n))
    random.shuffle(indices)

    n_val = max(1, int(n * args.val_split))
    val_indices = indices[:n_val]

    val_subset = Subset(val_dataset_raw, val_indices)
    print(f"Rebuilt validation split: {len(val_subset)} samples "
          f"(seed={args.seed}, val_split={args.val_split})")

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

    for loss_fn in ["edl", "softmax"]:
        print(f"\n=== Reliability diagram: {args.backbone} ({loss_fn}) ===")

        # Use the BEST checkpoint (peak val accuracy epoch), not latest
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

        save_path = os.path.join(
            args.output_dir, f"ham10000_{args.backbone}_{loss_fn}_reliability.png"
        )
        plot_reliability_diagram(
            bin_data, n_bins=args.n_bins, save_path=save_path,
            title=f"HAM10000 {args.backbone} -- {loss_fn} (ECE={ece:.4f}, {direction})"
        )
        print(f"  Saved reliability diagram -> {save_path}")

    print("\nDone. Compare the two saved PNGs.")


if __name__ == "__main__":
    main()