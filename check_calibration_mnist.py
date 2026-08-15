"""
Check calibration of the already-trained MNIST models.

Loads:
    1. EfficientNet-B0 + EDL checkpoint
    2. EfficientNet-B0 + Softmax checkpoint

Evaluates:
    - Test accuracy
    - Average confidence
    - ECE
    - Reliability diagram

This script DOES NOT train the models again.
It only loads the saved checkpoints and evaluates them.
"""

"""
Generates reliability diagrams from the checkpoints already trained by
validate_on_mnist.py -- no retraining needed. This is the visual explanation
for WHY EDL's ECE (0.0787) is higher than softmax's (0.0028) despite both
having ~99% accuracy: the diagram will show EDL's bars sitting BELOW the
diagonal (underconfident), rather than above it (overconfident) -- a
meaningfully different, and much less concerning, failure mode.

Usage:
    PYTHONPATH=. python check_calibration_mnist.py --backbone efficientnet_b0 \
        --checkpoint_dir /content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mnist_loader import get_mnist_datasets, MNIST_CLASSES
from losses.evidential_loss import edl_predictions
from metrics.ece import compute_ece, plot_reliability_diagram
from models.backbone_factory import get_backbone
from validate_on_mnist import checkpoint_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--checkpoint_dir", default="checkpoints_mnist_validation",
                    help="Must match the --checkpoint_dir used when running validate_on_mnist.py")
    p.add_argument("--mnist_root", default="data/mnist")
    p.add_argument("--output_dir", default="results/reliability_diagrams")
    return p.parse_args()


def load_trained_model(ckpt_path, backbone, num_classes, device):
    model = get_backbone(backbone, num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from {ckpt_path} (trained through epoch {ckpt['epoch']})")
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
    num_classes = len(MNIST_CLASSES)
    os.makedirs(args.output_dir, exist_ok=True)

    _, mnist_test = get_mnist_datasets(args.mnist_root)
    mnist_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    for loss_fn in ["edl", "softmax"]:
        print(f"\n=== Reliability diagram: {args.backbone} ({loss_fn}) ===")
        ckpt_path = checkpoint_path(args.checkpoint_dir, args.backbone, loss_fn)
        model = load_trained_model(ckpt_path, args.backbone, num_classes, device)

        confidences, predictions, labels = collect_predictions(model, mnist_loader, device, loss_fn)
        accuracy = float(np.mean(predictions == labels))
        avg_confidence = float(np.mean(confidences))
        ece, bin_data = compute_ece(confidences, predictions, labels, n_bins=args.n_bins)

        print(f"  accuracy={accuracy:.4f} | avg_confidence={avg_confidence:.4f} | ECE={ece:.4f}")
        direction = "UNDERconfident" if avg_confidence < accuracy else "OVERconfident"
        print(f"  Direction of miscalibration: {direction} "
              f"(avg confidence {avg_confidence:.4f} vs accuracy {accuracy:.4f})")

        save_path = os.path.join(args.output_dir, f"{args.backbone}_{loss_fn}_reliability.png")
        plot_reliability_diagram(
            bin_data, n_bins=args.n_bins, save_path=save_path,
            title=f"{args.backbone} -- {loss_fn} (ECE={ece:.4f}, {direction})"
        )
        print(f"  Saved reliability diagram -> {save_path}")

    print("\nDone. Compare the two saved PNGs: EDL's bars should sit BELOW the diagonal "
          "(underconfident) rather than scattered above/below randomly -- that confirms "
          "the ECE gap is a systematic, explainable effect of the KL-annealing term, not noise.")


if __name__ == "__main__":
    main()