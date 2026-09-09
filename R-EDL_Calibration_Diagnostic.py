"""
Extended calibration/confidence diagnostic: Standard EDL vs R-EDL (lambda=0.1),
on the same lesion-grouped validation split. Computes accuracy, overall avg
confidence, avg confidence split by correct/incorrect predictions, ECE, and
reliability diagrams for both methods -- to determine R-EDL's actual
miscalibration direction (over- vs under-confident) before deciding which
framework carries forward into the main factorial study.
"""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from datasets.ham10000 import HAM10000Dataset, default_transforms
from losses.evidential_loss import edl_predictions, redl_predictions
from metrics.ece import compute_ece, plot_reliability_diagram
from models.backbone_factory import get_backbone

CHECKPOINT_DIR = "/content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints"

DATA_ROOT = "/content/drive/MyDrive/Lightweight-EDL-Medical/data/ham10000"

OUTPUT_DIR = "/content/drive/MyDrive/Lightweight-EDL-Medical/results/reliability_diagrams"

SEED = 42
VAL_SPLIT = 0.15
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 7
REDL_LAMBDA = 0.1

CHECKPOINTS = {
    "edl": os.path.join(
        CHECKPOINT_DIR,
        f"ham10000_{BACKBONE}_edl_standard_lesion_seed{SEED}_best.pt",
    ),
    "redl": os.path.join(
        CHECKPOINT_DIR,
        f"ham10000_{BACKBONE}_redl_standard_lesion_seed{SEED}_lam{REDL_LAMBDA}_best.pt",
    ),
}


def build_val_loader():
    val_dataset_raw = HAM10000Dataset(DATA_ROOT, transform=default_transforms(train=False))
    n = len(val_dataset_raw)
    all_indices = np.arange(n)
    all_labels = np.array(val_dataset_raw.labels)
    all_lesion_ids = val_dataset_raw.metadata["lesion_id"].values

    n_splits = round(1 / VAL_SPLIT)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    _, val_indices = next(sgkf.split(all_indices, all_labels, groups=all_lesion_ids))

    val_subset = Subset(val_dataset_raw, val_indices)
    print(f"Validation split: {len(val_subset)} samples")
    return DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)


def load_model(ckpt_path, device):
    model = get_backbone(BACKBONE, num_classes=NUM_CLASSES, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {ckpt_path} (epoch {ckpt['epoch']}, best_metric={ckpt.get('best_metric')})")
    model.eval()
    return model


def collect_predictions(model, loader, device, loss_fn, lam=REDL_LAMBDA):
    all_conf, all_pred, all_label = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            if loss_fn == "edl":
                pred_class, confidence, _ = edl_predictions(output)
            elif loss_fn == "redl":
                pred_class, confidence, _ = redl_predictions(output, lam=lam)
            else:
                probs = torch.softmax(output, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
            all_conf.extend(confidence.cpu().numpy())
            all_pred.extend(pred_class.cpu().numpy())
            all_label.extend(labels.numpy())
    return np.array(all_conf), np.array(all_pred), np.array(all_label)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    val_loader = build_val_loader()
    results = []

    for loss_fn, ckpt_path in CHECKPOINTS.items():
        label = "Standard EDL" if loss_fn == "edl" else f"R-EDL (lambda={REDL_LAMBDA})"
        print(f"\n=== {label} ===")

        model = load_model(ckpt_path, device)
        confidences, predictions, labels = collect_predictions(model, val_loader, device, loss_fn)

        correct_mask = predictions == labels
        accuracy = float(np.mean(correct_mask))
        avg_confidence = float(np.mean(confidences))
        avg_conf_correct = float(np.mean(confidences[correct_mask])) if correct_mask.any() else float("nan")
        avg_conf_incorrect = float(np.mean(confidences[~correct_mask])) if (~correct_mask).any() else float("nan")

        ece, bin_data = compute_ece(confidences, predictions, labels, n_bins=15)

        print(f"  Accuracy                : {accuracy:.4f}")
        print(f"  Average confidence       : {avg_confidence:.4f}")
        print(f"  Avg confidence (correct) : {avg_conf_correct:.4f}")
        print(f"  Avg confidence (wrong)   : {avg_conf_incorrect:.4f}")
        print(f"  ECE                      : {ece:.4f}")

        direction = "UNDERconfident" if avg_confidence < accuracy else "OVERconfident"
        print(f"  Direction: {direction}")

        results.append((label, accuracy, avg_confidence, avg_conf_correct, avg_conf_incorrect, ece, direction))

        save_path = os.path.join(OUTPUT_DIR, f"ham10000_{loss_fn}_diagnostic_reliability.png")
        plot_reliability_diagram(
            bin_data, n_bins=15, save_path=save_path,
            title=f"{label} (ECE={ece:.4f}, {direction})"
        )
        print(f"  Saved -> {save_path}")

    print("\n=== Summary ===")
    print(f"{'Method':<20}{'Acc':>8}{'AvgConf':>10}{'ConfCorr':>10}{'ConfWrong':>11}{'ECE':>8}  Direction")
    for label, acc, conf, cc, cw, ece, direction in results:
        print(f"{label:<20}{acc:>8.4f}{conf:>10.4f}{cc:>10.4f}{cw:>11.4f}{ece:>8.4f}  {direction}")


if __name__ == "__main__":
    main()