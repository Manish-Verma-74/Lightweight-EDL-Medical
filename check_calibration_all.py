"""
Unified Calibration Diagnostic & Reliability Diagram Generator
Supports: Softmax, Standard EDL, R-EDL, and F-EDL (Controlled Adaptation).

Evaluates models on the exact lesion-grouped validation split, computing accuracy,
average confidence (overall, correct, wrong), ECE, and direction of miscalibration.
Generates both individual reliability diagrams and a 2x2 comparison grid figure.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from datasets.ham10000 import HAM10000Dataset, default_transforms
from losses.evidential_loss import edl_predictions, redl_predictions
from losses.fedl_loss import fedl_predictions
from metrics.ece import compute_ece, plot_reliability_diagram
from models.backbone_factory import get_backbone
from models.fedl_wrapper import FEDLWrapper

HAM10000_NUM_CLASSES = 7


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Calibration Diagnostic")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    parser.add_argument("--dataset", default="ham10000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--redl_lambda", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--data_root", default="data/ham10000")
    parser.add_argument("--output_dir", default="results/reliability_diagrams")
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
    print(f"Rebuilt lesion-level validation split: {len(val_subset)} samples "
          f"(seed={args.seed}, n_splits={n_splits})")

    return DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)


def get_checkpoint_path(args, loss_fn):
    """Explicitly maps each method to its actual completed experiment checkpoint filename."""
    if loss_fn == "softmax":
        filename = f"ham10000_{args.backbone}_softmax_standard_lesion_seed{args.seed}_best.pt"
    elif loss_fn == "edl":
        filename = f"ham10000_{args.backbone}_edl_standard_lesion_seed{args.seed}_best.pt"
    elif loss_fn == "redl":
        filename = f"ham10000_{args.backbone}_redl_standard_lesion_seed{args.seed}_lam{args.redl_lambda:g}_best.pt"
    elif loss_fn == "fedl":
        filename = f"ham10000_{args.backbone}_fedl_standard_seed{args.seed}_best.pt"
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    return os.path.join(args.checkpoint_dir, filename)


def load_model_for_method(ckpt_path, backbone, loss_fn, num_classes, device):
    """Instantiates model architecture using the exact method constructor interface."""
    if loss_fn == "fedl":
        model = FEDLWrapper(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=False
        ).to(device)
    else:
        model = get_backbone(
            backbone,
            num_classes=num_classes,
            pretrained=False
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    epoch = ckpt.get("epoch", "N/A")
    print(f"  Loaded: {os.path.basename(ckpt_path)} (Epoch {epoch})")
    model.eval()
    return model


def collect_predictions(model, loader, device, loss_fn, redl_lambda=0.1):
    """Extracts confidence, prediction, and labels using exact predictive distributions."""
    all_conf, all_pred, all_label = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)

            if loss_fn == "softmax":
                probs = torch.softmax(output, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)

            elif loss_fn == "edl":
                pred_class, confidence, _ = edl_predictions(output)

            elif loss_fn == "redl":
                pred_class, confidence, _ = redl_predictions(output, lam=redl_lambda)

            elif loss_fn == "fedl":
                alpha, p, tau = output
                pred_class, confidence, _, _, _ = fedl_predictions(alpha, p, tau)

            all_conf.extend(confidence.cpu().numpy())
            all_pred.extend(pred_class.cpu().numpy())
            all_label.extend(labels.numpy())

    return np.array(all_conf), np.array(all_pred), np.array(all_label)


def plot_2x2_comparative_grid(summary_dict, output_dir, backbone, n_bins=15):
    """Generates a uniform side-by-side 2x2 comparison plot for thesis inclusion."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=300)
    axes = axes.flatten()

    for idx, (method, data) in enumerate(summary_dict.items()):
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
        ax.bar(bin_lowers, bin_accs, width=widths, align="edge",
               edgecolor="black", color="#1f77b4", alpha=0.75)

        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(f"{method}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, f"ham10000_{backbone}_2x2_calibration_comparison.png")
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 2x2 Comparative Grid -> {grid_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    val_loader = build_val_loader(args)

    methods = [
        ("softmax", "Softmax Baseline"),
        ("edl", "Standard EDL"),
        ("redl", f"R-EDL (lambda={args.redl_lambda})"),
        ("fedl", "F-EDL (Controlled Adaptation)"),
    ]

    results_table = []
    comparative_data = {}

    for loss_fn, display_name in methods:
        print(f"\n=== Evaluating: {display_name} ===")
        ckpt_path = get_checkpoint_path(args, loss_fn)

        if not os.path.exists(ckpt_path):
            print(f"❌ Error: Expected checkpoint not found at {ckpt_path}")
            continue

        model = load_model_for_method(ckpt_path, args.backbone, loss_fn, HAM10000_NUM_CLASSES, device)
        confidences, predictions, labels = collect_predictions(
            model, val_loader, device, loss_fn, redl_lambda=args.redl_lambda
        )

        correct_mask = predictions == labels
        accuracy = float(np.mean(correct_mask))
        avg_conf = float(np.mean(confidences))
        avg_conf_corr = float(np.mean(confidences[correct_mask])) if correct_mask.any() else float("nan")
        avg_conf_wrong = float(np.mean(confidences[~correct_mask])) if (~correct_mask).any() else float("nan")

        ece, bin_data = compute_ece(confidences, predictions, labels, n_bins=args.n_bins)
        direction = "UNDERconfident" if avg_conf < accuracy else "OVERconfident"

        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Avg Conf : {avg_conf:.4f} (Correct: {avg_conf_corr:.4f} | Wrong: {avg_conf_wrong:.4f})")
        print(f"  ECE      : {ece:.4f} | Diagnostic Direction: {direction}")

        results_table.append((display_name, accuracy, avg_conf, avg_conf_corr, avg_conf_wrong, ece, direction))

        comparative_data[display_name] = {
            "confidences": confidences,
            "predictions": predictions,
            "labels": labels,
            "ece": ece,
        }

        save_path = os.path.join(args.output_dir, f"ham10000_{args.backbone}_{loss_fn}_reliability.png")
        plot_reliability_diagram(
            bin_data, n_bins=args.n_bins, save_path=save_path,
            title=f"{display_name} (ECE={ece:.4f})"
        )

    print("\n" + "=" * 85)
    print("FINAL CALIBRATION DIAGNOSTIC SUMMARY")
    print("=" * 85)
    print(f"{'Framework':<30}{'Acc':>8}{'AvgConf':>10}{'ConfCorr':>10}{'ConfWrong':>11}{'ECE':>8}  Direction")
    print("-" * 85)
    for label, acc, conf, cc, cw, ece, direction in results_table:
        print(f"{label:<30}{acc:>8.4f}{conf:>10.4f}{cc:>10.4f}{cw:>11.4f}{ece:>8.4f}  {direction}")
    print("=" * 85)

    if len(comparative_data) > 1:
        plot_2x2_comparative_grid(comparative_data, args.output_dir, args.backbone, n_bins=args.n_bins)


if __name__ == "__main__":
    main()