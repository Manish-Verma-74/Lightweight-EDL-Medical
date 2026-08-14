"""
Validates our EDL loss and ECE implementation by training the same
EfficientNet-B0 backbone on MNIST using both EDL and standard Softmax
cross-entropy, then comparing their test accuracy and ECE.

Sanity-validate the EDL loss (losses/evidential_loss.py) and ECE metric
(metrics/ece.py) against real training dynamics, by training one of our
lightweight backbones on MNIST two ways:

    1. EDL loss (edl_mse_loss)          -- our thesis approach
    2. Plain softmax cross-entropy      -- standard baseline

We then compare test accuracy AND test ECE between the two. The expected
pattern, consistent with Sensoy et al. (2018) and Guo et al. (2017):
    - Both should reach similar, high accuracy on MNIST (it's an easy dataset)
    - EDL should show equal-or-better calibration (lower or comparable ECE)
      than plain softmax cross-entropy, since softmax classifiers are known
      to be overconfident.

This is a validation/sanity-check script, NOT part of the main HAM10000
thesis experiments -- MNIST + these specific backbones is just a fast,
reliable way to confirm the loss and metric code behave correctly on real
(not synthetic) training data before trusting them on the real medical task.

Usage:
    PYTHONPATH=. python validate_on_mnist.py --backbone efficientnet_b0 --epochs 3
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets.mnist_loader import get_mnist_datasets, MNIST_CLASSES
from losses.evidential_loss import edl_mse_loss, edl_predictions
from metrics.ece import compute_ece
from models.backbone_factory import get_backbone


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--annealing_step", type=int, default=3)
    p.add_argument("--debug_subset", type=int, default=None,
                    help="If set, use only N train / N test samples (fast dry run)")
    p.add_argument("--data_root", default="data/mnist")
    return p.parse_args()


def train_one_model(model, loader, optimizer, loss_fn, epochs, device, num_classes, annealing_step):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            if loss_fn == "edl":
                loss = edl_mse_loss(output, labels, epoch_num=epoch, num_classes=num_classes,
                                     annealing_step=annealing_step, device=device)
            else:
                loss = nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(loader.dataset)
        print(f"  [{loss_fn}] epoch {epoch}/{epochs} | loss={avg_loss:.4f} | {time.time()-start:.1f}s")


def evaluate(model, loader, device, loss_fn):
    model.eval()
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

    all_pred, all_label = np.array(all_pred), np.array(all_label)
    accuracy = float(np.mean(all_pred == all_label))
    ece, _ = compute_ece(all_conf, all_pred, all_label, n_bins=15)
    return accuracy, ece


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = len(MNIST_CLASSES)

    train_ds, test_ds = get_mnist_datasets(args.data_root)
    if args.debug_subset:
        train_ds = Subset(train_ds, range(args.debug_subset))
        test_ds = Subset(test_ds, range(min(args.debug_subset, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    results = {}
    for loss_fn in ["edl", "softmax"]:
        print(f"\n=== Training {args.backbone} with {loss_fn} loss ===")
        model = get_backbone(args.backbone, num_classes=num_classes, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_one_model(model, train_loader, optimizer, loss_fn, args.epochs, device,
                         num_classes, args.annealing_step)
        accuracy, ece = evaluate(model, test_loader, device, loss_fn)
        results[loss_fn] = (accuracy, ece)
        print(f"  -> test accuracy={accuracy:.4f} | test ECE={ece:.4f}")

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"{'Loss':<10} {'Accuracy':<12} {'ECE':<10}")
    for loss_fn, (acc, ece) in results.items():
        print(f"{loss_fn:<10} {acc:<12.4f} {ece:<10.4f}")

    edl_acc, edl_ece = results["edl"]
    sm_acc, sm_ece = results["softmax"]
    print("\nSanity checks:")
    print(f"  - Both reach high accuracy on MNIST (expected, it's an easy dataset): "
          f"{'PASS' if edl_acc > 0.9 and sm_acc > 0.9 else 'CHECK -- accuracy lower than expected'}")
    print(f"  - EDL calibration (ECE) is comparable to or better than softmax "
          f"(Sensoy 2018 / Guo 2017 expectation): "
          f"{'PASS' if edl_ece <= sm_ece + 0.02 else 'CHECK -- EDL ECE notably worse than softmax'}")


if __name__ == "__main__":
    main()