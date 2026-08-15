"""
Out-of-distribution (OOD) validation: the REAL test of what EDL is supposed
to do, per Sensoy et al. (2018) -- reuses the checkpoints already trained by
validate_on_mnist.py, no retraining needed.

We evaluate both the EDL model and the softmax model on:
    - IN-DISTRIBUTION data: MNIST test set (what they were trained on)
    - OUT-OF-DISTRIBUTION data: Fashion-MNIST test set (never seen, different content)

Expected pattern (this is the actual claim the papers make, more so than
in-distribution ECE):
    - Both models should be confident on MNIST (in-distribution)
    - On Fashion-MNIST (OOD), a well-behaved EDL model's confidence should
      drop substantially, and its uncertainty (K/S) should rise -- because it
      has almost no "evidence" for any class on inputs unlike its training data
    - A plain softmax model has no such mechanism -- softmax always produces
      SOME confident-looking distribution, even on nonsense input, so its
      confidence typically stays high on OOD data too. This is the core
      failure mode EDL exists to fix.

Usage:
    PYTHONPATH=. python ood_validation.py --backbone efficientnet_b0 \
        --checkpoint_dir /content/drive/MyDrive/Lightweight-EDL-Medical/checkpoints
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mnist_loader import get_mnist_datasets, MNIST_CLASSES
from datasets.fashion_mnist_loader import get_fashion_mnist_test_set
from losses.evidential_loss import edl_predictions
from models.backbone_factory import get_backbone
from validate_on_mnist import checkpoint_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="efficientnet_b0",
                    choices=["efficientnet_b0", "mobilenet_v3_small", "shufflenet_v2"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--checkpoint_dir", default="checkpoints_mnist_validation",
                    help="Must match the --checkpoint_dir used when running validate_on_mnist.py")
    p.add_argument("--mnist_root", default="data/mnist")
    p.add_argument("--fashion_root", default="data/fashion_mnist")
    return p.parse_args()


def load_trained_model(ckpt_path, backbone, num_classes, device):
    model = get_backbone(backbone, num_classes=num_classes, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from {ckpt_path} (trained through epoch {ckpt['epoch']})")
    model.eval()
    return model


def collect_confidence_and_uncertainty(model, loader, device, loss_fn):
    """Returns (avg_confidence, avg_uncertainty). avg_uncertainty is only meaningful for EDL;
    for softmax we report 1 - confidence as a rough proxy since softmax has no native
    uncertainty measure."""
    all_conf, all_unc = [], []
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            output = model(images)
            if loss_fn == "edl":
                _pred, confidence, uncertainty = edl_predictions(output)
                all_unc.extend(uncertainty.cpu().numpy())
            else:
                probs = torch.softmax(output, dim=1)
                confidence, _pred = torch.max(probs, dim=1)
                all_unc.extend((1 - confidence).cpu().numpy())
            all_conf.extend(confidence.cpu().numpy())
    return float(np.mean(all_conf)), float(np.mean(all_unc))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = len(MNIST_CLASSES)

    print("\nLoading in-distribution (MNIST) test set...")
    _, mnist_test = get_mnist_datasets(args.mnist_root)
    mnist_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("Loading out-of-distribution (Fashion-MNIST) test set...")
    fashion_test = get_fashion_mnist_test_set(args.fashion_root)
    fashion_loader = DataLoader(fashion_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}
    for loss_fn in ["edl", "softmax"]:
        print(f"\n=== Evaluating {args.backbone} ({loss_fn}) ===")
        ckpt_path = checkpoint_path(args.checkpoint_dir, args.backbone, loss_fn)
        model = load_trained_model(ckpt_path, args.backbone, num_classes, device)

        id_conf, id_unc = collect_confidence_and_uncertainty(model, mnist_loader, device, loss_fn)
        ood_conf, ood_unc = collect_confidence_and_uncertainty(model, fashion_loader, device, loss_fn)

        print(f"  In-distribution  (MNIST)         -> avg confidence={id_conf:.4f}, avg uncertainty={id_unc:.4f}")
        print(f"  Out-of-distribution (Fashion-MNIST) -> avg confidence={ood_conf:.4f}, avg uncertainty={ood_unc:.4f}")
        print(f"  Confidence drop on OOD: {id_conf - ood_conf:+.4f}")

        results[loss_fn] = {
            "id_conf": id_conf, "ood_conf": ood_conf,
            "id_unc": id_unc, "ood_unc": ood_unc,
            "conf_drop": id_conf - ood_conf,
        }

    print("\n" + "=" * 60)
    print("OOD VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Loss':<10} {'ID conf':<10} {'OOD conf':<10} {'Conf drop':<12}")
    for loss_fn, r in results.items():
        print(f"{loss_fn:<10} {r['id_conf']:<10.4f} {r['ood_conf']:<10.4f} {r['conf_drop']:<+12.4f}")

    edl_drop = results["edl"]["conf_drop"]
    sm_drop = results["softmax"]["conf_drop"]
    print("\nSanity check:")
    print(f"  - EDL's confidence drop on OOD data is larger than softmax's "
          f"(EDL is expected to reduce confidence on OOD inputs compared with "
          f"standard softmax classifiers): "
          f"{'PASS' if edl_drop > sm_drop else 'CHECK -- EDL does not show a larger OOD confidence drop than softmax'}")


if __name__ == "__main__":
    main()