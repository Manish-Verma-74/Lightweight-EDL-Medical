"""
Phase 2: Dedicated F-EDL Backbone Calibration & Uncertainty Diagnostic
Evaluates EfficientNet-B0, ShuffleNetV2, and MobileNetV3-Small on the exact
lesion-grouped validation split.
"""

import argparse
import inspect
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from datasets.ham10000 import HAM10000Dataset, default_transforms
from losses.fedl_loss import fedl_predictions
from metrics.ece import compute_ece
from models.fedl_wrapper import FEDLWrapper

HAM10000_NUM_CLASSES = 7


def parse_args():
    parser = argparse.ArgumentParser(description="F-EDL Backbone Study Diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_root", default="data/ham10000")
    parser.add_argument("--output_dir", default="results/backbone_study")
    return parser.parse_args()


def build_val_loader(args):
    """Rebuilds the exact lesion-grouped validation split used during training."""
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
    print(f"Rebuilt lesion-level validation split: {len(val_subset)} samples (seed={args.seed})")

    return DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)


def collect_fedl_predictions(model, loader, device):
    """Extracts predictions, confidence, and evidential uncertainty components."""
    all_conf, all_pred, all_label = [], [], []
    all_tu, all_eu, all_au = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            alpha, p, tau = model(images)
            
            # Unpack in the exact return order of fedl_predictions: (pred_class, confidence, tu, eu, au)
            pred_class, confidence, tu, eu, au = fedl_predictions(alpha, p, tau)

            all_conf.extend(confidence.cpu().numpy())
            all_pred.extend(pred_class.cpu().numpy())
            all_label.extend(labels.numpy())
            all_tu.extend(tu.cpu().numpy())
            all_eu.extend(eu.cpu().numpy())
            all_au.extend(au.cpu().numpy())

    return (
        np.array(all_conf),
        np.array(all_pred),
        np.array(all_label),
        np.array(all_tu),
        np.array(all_eu),
        np.array(all_au),
    )


def safe_compute_ece(conf, pred, label, n_bins):
    """Handles potential variations in the compute_ece return signature safely."""
    sig = inspect.signature(compute_ece)
    params = sig.parameters
    
    # Check if function expects targets as 'targets' or 'labels' or positional
    res = compute_ece(conf, pred, label, n_bins=n_bins)
    if isinstance(res, tuple):
        return res[0]
    return res


def plot_3panel_reliability_grid(summary_dict, output_dir, n_bins=15):
    """Generates a 1x3 comparative reliability plot across backbones."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    for idx, (display_name, data) in enumerate(summary_dict.items()):
        ax = axes[idx]
        confidences = data["confidences"]
        predictions = data["predictions"]
        labels = data["labels"]
        ece = data["ece"]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accs = []
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            if in_bin.any():
                bin_accs.append(np.mean(predictions[in_bin] == labels[in_bin]))
            else:
                bin_accs.append(np.nan)

        widths = bin_uppers - bin_lowers

        ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal Calibration")
        ax.bar(
            bin_lowers,
            bin_accs,
            width=widths,
            align="edge",
            edgecolor="black",
            color="#1f77b4",
            alpha=0.75,
        )

        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(f"{display_name}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "fedl_backbones_reliability_comparison.png")
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 1x3 Backbone Grid -> {grid_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_loader = build_val_loader(args)

    backbones = [
        ("efficientnet_b0", "EfficientNet-B0", f"ham10000_efficientnet_b0_fedl_standard_seed{args.seed}_best.pt"),
        ("shufflenet_v2", "ShuffleNetV2", f"ham10000_shufflenet_v2_fedl_standard_seed{args.seed}_best.pt"),
        ("mobilenet_v3_small", "MobileNetV3-Small", f"ham10000_mobilenet_v3_small_fedl_standard_seed{args.seed}_best.pt"),
    ]

    results_table = []
    comparative_data = {}

    for bb_key, display_name, ckpt_file in backbones:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)
        print(f"\n=== Evaluating Backbone: {display_name} ===")

        if not os.path.exists(ckpt_path):
            print(f"❌ Error: Checkpoint not found at {ckpt_path}")
            continue

        model = FEDLWrapper(backbone_name=bb_key, num_classes=HAM10000_NUM_CLASSES, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        conf, pred, label, tu, eu, au = collect_fedl_predictions(model, val_loader, device)

        correct_mask = pred == label
        acc = float(np.mean(correct_mask))
        avg_conf = float(np.mean(conf))
        avg_conf_corr = float(np.mean(conf[correct_mask])) if correct_mask.any() else float("nan")
        avg_conf_wrong = float(np.mean(conf[~correct_mask])) if (~correct_mask).any() else float("nan")

        mean_tu = float(np.mean(tu))
        mean_eu = float(np.mean(eu))
        mean_au = float(np.mean(au))

        ece = safe_compute_ece(conf, pred, label, n_bins=args.n_bins)

        print(f"  Accuracy : {acc:.4f} | ECE: {ece:.4f}")
        print(f"  Avg Conf : {avg_conf:.4f} (Correct: {avg_conf_corr:.4f} | Wrong: {avg_conf_wrong:.4f})")
        print(f"  Uncertainty -> Total: {mean_tu:.4f} | Epistemic: {mean_eu:.4f} | Aleatoric: {mean_au:.4f}")

        results_table.append((display_name, acc, ece, avg_conf, avg_conf_corr, avg_conf_wrong, mean_tu, mean_eu, mean_au))

        comparative_data[display_name] = {
            "confidences": conf,
            "predictions": pred,
            "labels": label,
            "ece": ece,
        }

    print("\n" + "=" * 105)
    print("PHASE 2: F-EDL BACKBONE DIAGNOSTIC & UNCERTAINTY SUMMARY")
    print("=" * 105)
    print(f"{'Backbone':<20}{'Acc':>8}{'ECE':>8}{'AvgConf':>9}{'ConfCorr':>10}{'ConfWrong':>11}{'TU':>8}{'EU':>8}{'AU':>8}")
    print("-" * 105)
    for row in results_table:
        name, acc, ece, conf, cc, cw, tu, eu, au = row
        print(f"{name:<20}{acc:>8.4f}{ece:>8.4f}{conf:>9.4f}{cc:>10.4f}{cw:>11.4f}{tu:>8.4f}{eu:>8.4f}{au:>8.4f}")
    print("=" * 105)

    if len(comparative_data) > 1:
        plot_3panel_reliability_grid(comparative_data, args.output_dir, n_bins=args.n_bins)


if __name__ == "__main__":
    main()