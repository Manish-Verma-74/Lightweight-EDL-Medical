"""
HAM10000 skin lesion dataset loader.

Expects the Kaggle HAM10000 layout after download, e.g.:
    data/ham10000/
        HAM10000_metadata.csv
        HAM10000_images_part_1/
        HAM10000_images_part_2/

Download (run once, requires kaggle.json API credentials configured):
    !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000 --unzip

Usage as a script (verification):
    PYTHONPATH=. python datasets/ham10000.py data/ham10000
"""
import os
import sys

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class HAM10000Dataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        meta_path = os.path.join(root_dir, "HAM10000_metadata.csv")
        self.metadata = pd.read_csv(meta_path)

        # Both image folders may exist; build an image_id -> filepath lookup
        self.image_paths = {}
        for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            folder = os.path.join(root_dir, sub)
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith(".jpg"):
                        image_id = os.path.splitext(fname)[0]
                        self.image_paths[image_id] = os.path.join(folder, fname)

        # Keep only rows whose image was actually found on disk
        self.metadata = self.metadata[self.metadata["image_id"].isin(self.image_paths.keys())].reset_index(drop=True)
        self.labels = self.metadata["dx"].map(CLASS_TO_IDX).values

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.image_paths[row["image_id"]]
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler to counteract HAM10000's heavy class imbalance (nv dominates)."""
        class_counts = pd.Series(self.labels).value_counts().sort_index()
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in self.labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def default_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/ham10000"
    ds = HAM10000Dataset(root, transform=default_transforms(train=False))
    print(f"Loaded {len(ds)} samples across {len(CLASS_NAMES)} classes")
    print("Class distribution:", pd.Series(ds.labels).value_counts().sort_index().to_dict())

    if len(ds) > 0:
        # Save a small sample grid to verify images/labels line up
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            img, label = ds[i]
            img_disp = img.permute(1, 2, 0).numpy()
            img_disp = (img_disp * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            ax.imshow(img_disp.clip(0, 1))
            ax.set_title(CLASS_NAMES[label])
            ax.axis("off")
        fig.tight_layout()
        out_path = os.path.join(root, "sample_grid.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved sample grid to {out_path}")
    else:
        print("No samples found -- check that the dataset was downloaded and extracted correctly.")
