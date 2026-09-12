# ============================================================
# STAGE 4 — FULL OOD DETECTION
# 3 Backbones × 4 Frameworks = 12 Configurations
#
# ID      : HAM10000 validation (lesion-level)
# Near-OOD: PAD-UFES-20
# Far-OOD : CIFAR-100
#
# Metrics:
#   ID  : Accuracy, ECE, Mean ID Uncertainty
#   OOD : AUROC, FPR95, AUPR-In, AUPR-Out
#   OOD : Mean Near-OOD Uncertainty
#         Mean Far-OOD Uncertainty
#   Derived:
#         ΔU(Near-ID), ΔU(Far-ID)
#
# Restart-safe:
#   - Saves CSV after every configuration
#   - Saves raw uncertainty scores
#   - Skips completed configurations
#   - num_workers=0 to avoid Colab worker issues
# ============================================================

import os
import sys
import json
import gc
import time
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT / DEVICE SETUP
# ============================================================

PROJECT_ROOT = "/content/Lightweight-EDL-Medical"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STAGE 4 — FULL OOD DETECTION")
print("=" * 70)
print("Project :", PROJECT_ROOT)
print("Device  :", DEVICE)

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))

print()


# ============================================================
# 2. PROJECT IMPORTS
# ============================================================

from models.fedl_wrapper import FEDLWrapper
from models.backbone_factory import get_backbone
from metrics.ece import compute_ece

print("✓ Project imports successful")


# ============================================================
# 3. PATHS
# ============================================================

HAM_ROOT = "/content/data/ham10000"
PAD_ROOT = "/content/data/pad_ufes20"
CIFAR_ROOT = "/content/data/cifar100"

CHECKPOINT_DIR = (
    "/content/drive/MyDrive/"
    "Lightweight-EDL-Medical/checkpoints"
)

RESULT_DIR = (
    "/content/drive/MyDrive/"
    "Lightweight-EDL-Medical/results/ood_stage4"
)

RAW_SCORE_DIR = os.path.join(RESULT_DIR, "raw_scores")

CSV_PATH = os.path.join(
    RESULT_DIR,
    "stage4_full_ood_results.csv"
)

STATUS_PATH = os.path.join(
    RESULT_DIR,
    "stage4_status.json"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RAW_SCORE_DIR, exist_ok=True)

print("✓ Result directories ready")
print("CSV:", CSV_PATH)
print()


# ============================================================
# 4. CONSTANTS
# ============================================================

NUM_CLASSES = 7
BATCH_SIZE = 64
IMAGE_SIZE = 224
SEED = 42

BACKBONES = [
    "efficientnet_b0",
    "mobilenet_v3_small",
    "shufflenet_v2",
]

FRAMEWORKS = [
    "softmax",
    "edl",
    "redl",
    "fedl",
]

REDL_LAMBDA = 0.1


# ============================================================
# 5. EXACT CHECKPOINT LOOKUP
# ============================================================

CHECKPOINT_LOOKUP = {

    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    ("efficientnet_b0", "edl"):
        "ham10000_efficientnet_b0_edl_standard_lesion_seed42_best.pt",

    ("efficientnet_b0", "softmax"):
        "ham10000_efficientnet_b0_softmax_standard_lesion_seed42_best.pt",

    ("efficientnet_b0", "redl"):
        "ham10000_efficientnet_b0_redl_standard_lesion_seed42_lam0.1_best.pt",

    ("efficientnet_b0", "fedl"):
        "ham10000_efficientnet_b0_fedl_standard_seed42_best.pt",


    # --------------------------------------------------------
    # MobileNetV3-Small
    # --------------------------------------------------------

    ("mobilenet_v3_small", "edl"):
        "ham10000_mobilenet_v3_small_edl_standard_seed42_best.pt",

    ("mobilenet_v3_small", "softmax"):
        "ham10000_mobilenet_v3_small_softmax_standard_seed42_best.pt",

    ("mobilenet_v3_small", "redl"):
        "ham10000_mobilenet_v3_small_redl_standard_seed42_lam0.1_best.pt",

    ("mobilenet_v3_small", "fedl"):
        "ham10000_mobilenet_v3_small_fedl_standard_seed42_best.pt",


    # --------------------------------------------------------
    # ShuffleNetV2
    # --------------------------------------------------------

    ("shufflenet_v2", "edl"):
        "ham10000_shufflenet_v2_edl_standard_seed42_best.pt",

    ("shufflenet_v2", "softmax"):
        "ham10000_shufflenet_v2_softmax_standard_seed42_best.pt",

    ("shufflenet_v2", "redl"):
        "ham10000_shufflenet_v2_redl_standard_seed42_lam0.1_best.pt",

    ("shufflenet_v2", "fedl"):
        "ham10000_shufflenet_v2_fedl_standard_seed42_best.pt",
}


# ============================================================
# 6. VERIFY ALL CHECKPOINTS BEFORE STARTING
# ============================================================

print("Checking 12 checkpoints...\n")

missing_checkpoints = []

for key, filename in CHECKPOINT_LOOKUP.items():

    path = os.path.join(CHECKPOINT_DIR, filename)

    if os.path.exists(path):
        print("✓", key, "->", filename)
    else:
        print("✗ MISSING:", key, "->", path)
        missing_checkpoints.append((key, path))

if missing_checkpoints:

    print("\nERROR: Missing checkpoint(s).")
    print("Fix the missing files before running Stage 4.")

    raise FileNotFoundError(
        f"{len(missing_checkpoints)} checkpoint(s) missing."
    )

print("\n✓ All 12 checkpoints found\n")


# ============================================================
# 7. HAM10000 ID DATASET
# ============================================================

print("=" * 70)
print("BUILDING HAM10000 ID VALIDATION SET")
print("=" * 70)

import importlib.util

HAM_FILE = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "ham10000.py"
)

spec = importlib.util.spec_from_file_location(
    "ham10000_stage4",
    HAM_FILE
)

ham_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ham_module)

HAM10000Dataset = ham_module.HAM10000Dataset
default_transforms = ham_module.default_transforms


full_dataset = HAM10000Dataset(
    HAM_ROOT,
    transform=default_transforms(train=False)
)

metadata = full_dataset.metadata.reset_index(drop=True)

X = np.arange(len(metadata))
y = metadata["dx"].values
groups = metadata["lesion_id"].values

from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    sgkf.split(X, y, groups)
)

val_dataset = Subset(
    full_dataset,
    val_idx
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("HAM10000 total :", len(full_dataset))
print("HAM10000 ID    :", len(val_dataset))
print("ID batches    :", len(val_loader))
print()


# ============================================================
# 8. PAD-UFES-20 DATASET
# ============================================================

print("=" * 70)
print("BUILDING PAD-UFES-20 NEAR-OOD SET")
print("=" * 70)


class PADUFES20Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        metadata_path = os.path.join(
            root_dir,
            "metadata.csv"
        )

        self.metadata = pd.read_csv(
            metadata_path
        )

        self.image_paths = {}

        images_root = os.path.join(
            root_dir,
            "images"
        )

        for part in [
            "imgs_part_1",
            "imgs_part_2",
            "imgs_part_3"
        ]:

            folder = os.path.join(
                images_root,
                part
            )

            if os.path.isdir(folder):

                for fname in os.listdir(folder):

                    if fname.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ):

                        self.image_paths[fname] = os.path.join(
                            folder,
                            fname
                        )

        self.metadata = self.metadata[
            self.metadata["img_id"].isin(
                self.image_paths.keys()
            )
        ].reset_index(drop=True)


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):

        from PIL import Image

        row = self.metadata.iloc[idx]

        image = Image.open(
            self.image_paths[row["img_id"]]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


ood_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


pad_dataset = PADUFES20Dataset(
    PAD_ROOT,
    transform=ood_transform
)

pad_loader = DataLoader(
    pad_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("PAD-UFES metadata :", len(pad_dataset.metadata))
print("PAD-UFES samples  :", len(pad_dataset))
print("PAD-UFES batches  :", len(pad_loader))
print()


# ============================================================
# 9. CIFAR-100 FAR-OOD DATASET
# ============================================================

print("=" * 70)
print("BUILDING CIFAR-100 FAR-OOD SET")
print("=" * 70)


cifar_dataset = CIFAR100(
    root=CIFAR_ROOT,
    train=False,
    download=False,
    transform=ood_transform
)

cifar_loader = DataLoader(
    cifar_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("CIFAR-100 samples :", len(cifar_dataset))
print("CIFAR-100 batches :", len(cifar_loader))
print()


# ============================================================
# 10. FINAL LOADER SANITY CHECK
# ============================================================

print("=" * 70)
print("FINAL STAGE 4 DATA LOADER SANITY CHECK")
print("=" * 70)

print(
    f"HAM10000 ID       : {len(val_dataset)} samples | "
    f"{len(val_loader)} batches"
)

print(
    f"PAD-UFES-20 Near  : {len(pad_dataset)} samples | "
    f"{len(pad_loader)} batches"
)

print(
    f"CIFAR-100 Far     : {len(cifar_dataset)} samples | "
    f"{len(cifar_loader)} batches"
)

assert len(val_dataset) == 1441
assert len(pad_dataset) == 2298
assert len(cifar_dataset) == 10000

print("\n✓ All Stage 4 loader counts verified")
print()


# ============================================================
# 11. MODEL CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Raw state_dict
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        print(
            "WARNING — missing keys:",
            len(missing)
        )

    if unexpected:
        print(
            "WARNING — unexpected keys:",
            len(unexpected)
        )

    return model


# ============================================================
# 12. MODEL BUILDER
# ============================================================

def build_model(backbone, framework):

    if framework == "fedl":

        model = FEDLWrapper(
            backbone_name=backbone,
            num_classes=NUM_CLASSES,
            pretrained=False
        )

    else:

        model = get_backbone(
            backbone,
            num_classes=NUM_CLASSES,
            pretrained=False
        )

    model = model.to(DEVICE)

    return model


# ============================================================
# 13. OUTPUT → PROBABILITY + UNCERTAINTY
# ============================================================

def model_output_to_predictions(
    output,
    framework
):

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    if framework == "softmax":

        logits = output

        prob = F.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            prob,
            dim=1
        )

        uncertainty = 1.0 - confidence

        return (
            prediction,
            confidence,
            uncertainty
        )


    # --------------------------------------------------------
    # Standard EDL
    # --------------------------------------------------------

    elif framework == "edl":

        logits = output

        evidence = F.relu(logits)

        alpha = evidence + 1.0

        S = torch.sum(
            alpha,
            dim=1,
            keepdim=True
        )

        prob = alpha / S

        confidence, prediction = torch.max(
            prob,
            dim=1
        )

        uncertainty = (
            NUM_CLASSES / S
        ).squeeze(1)

        return (
            prediction,
            confidence,
            uncertainty
        )


    # --------------------------------------------------------
    # R-EDL
    # --------------------------------------------------------

    elif framework == "redl":

        logits = output

        evidence = F.relu(logits)

        alpha = evidence + REDL_LAMBDA

        S = torch.sum(
            alpha,
            dim=1,
            keepdim=True
        )

        prob = alpha / S

        confidence, prediction = torch.max(
            prob,
            dim=1
        )

        # Correct R-EDL uncertainty:
        # u = lambda * K / S

        uncertainty = (
            REDL_LAMBDA * NUM_CLASSES / S
        ).squeeze(1)

        return (
            prediction,
            confidence,
            uncertainty
        )


    # --------------------------------------------------------
    # F-EDL
    # --------------------------------------------------------

    elif framework == "fedl":

        # Verified repository output:
        #
        # output[0] = alpha [B,K]
        # output[1] = p     [B,K]
        # output[2] = tau   [B,1]

        alpha, p, tau = output

        alpha0 = alpha.sum(
            dim=1,
            keepdim=True
        )

        denominator = (
            alpha0 + tau
        )

        # Expected probability
        mu = (
            alpha + tau * p
        ) / denominator

        # F-EDL predictive variance
        variance = (
            mu * (1.0 - mu)
            / (denominator + 1.0)
            +
            (
                tau ** 2
            )
            * p
            * (1.0 - p)
            /
            (
                denominator
                * (denominator + 1.0)
            )
        )

        confidence, prediction = torch.max(
            mu,
            dim=1
        )

        # Total uncertainty = sum of label-wise variance
        uncertainty = variance.sum(
            dim=1
        )

        return (
            prediction,
            confidence,
            uncertainty
        )


    else:

        raise ValueError(
            f"Unknown framework: {framework}"
        )


# ============================================================
# 14. ID EVALUATION
# ============================================================

def evaluate_id(
    model,
    framework,
    loader
):

    all_predictions = []
    all_confidences = []
    all_uncertainties = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            output = model(images)

            prediction, confidence, uncertainty = (
                model_output_to_predictions(
                    output,
                    framework
                )
            )

            all_predictions.append(
                prediction.detach().cpu().numpy()
            )

            all_confidences.append(
                confidence.detach().cpu().numpy()
            )

            all_uncertainties.append(
                uncertainty.detach().cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )


    predictions = np.concatenate(
        all_predictions
    )

    confidences = np.concatenate(
        all_confidences
    )

    uncertainties = np.concatenate(
        all_uncertainties
    )

    labels = np.concatenate(
        all_labels
    )


    accuracy = float(
        np.mean(
            predictions == labels
        )
    )


    # Exact project ECE implementation
    ece_result = compute_ece(
        confidences,
        predictions,
        labels,
        n_bins=15
    )

    if isinstance(ece_result, tuple):

        ece = float(
            ece_result[0]
        )

    else:

        ece = float(
            ece_result
        )


    mean_uncertainty = float(
        np.mean(uncertainties)
    )


    return {
        "accuracy": accuracy,
        "ece": ece,
        "mean_uncertainty": mean_uncertainty,
        "uncertainties": uncertainties,
        "predictions": predictions,
        "confidences": confidences,
        "labels": labels,
    }


# ============================================================
# 15. OOD SCORE COLLECTION
# ============================================================

def collect_ood_uncertainty(
    model,
    framework,
    loader
):

    all_uncertainties = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            # Dataset returns:
            # PAD -> image
            # CIFAR -> (image, label)

            if isinstance(batch, (tuple, list)):

                images = batch[0]

            else:

                images = batch

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            output = model(images)

            (
                prediction,
                confidence,
                uncertainty
            ) = model_output_to_predictions(
                output,
                framework
            )

            all_uncertainties.append(
                uncertainty.detach().cpu().numpy()
            )


    return np.concatenate(
        all_uncertainties
    )


# ============================================================
# 16. FPR95
# ============================================================

def compute_fpr95(
    id_scores,
    ood_scores
):

    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])

    scores = np.concatenate([
        id_scores,
        ood_scores
    ])

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    valid = np.where(
        tpr >= 0.95
    )[0]

    if len(valid) == 0:
        return float("nan")

    # First point where TPR reaches 95%
    idx = valid[0]

    return float(
        fpr[idx]
    )


# ============================================================
# 17. COMPLETE OOD METRICS
# ============================================================

def compute_ood_metrics(
    id_uncertainty,
    ood_uncertainty
):

    labels = np.concatenate([
        np.zeros(len(id_uncertainty)),
        np.ones(len(ood_uncertainty))
    ])

    scores = np.concatenate([
        id_uncertainty,
        ood_uncertainty
    ])


    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auroc = float(
        roc_auc_score(
            labels,
            scores
        )
    )


    # --------------------------------------------------------
    # AUPR-Out
    #
    # OOD = positive
    # uncertainty = score
    # --------------------------------------------------------

    aupr_out = float(
        average_precision_score(
            labels,
            scores
        )
    )


    # --------------------------------------------------------
    # AUPR-In
    #
    # ID = positive
    # confidence-like score = 1 - uncertainty
    # --------------------------------------------------------

    id_labels = 1.0 - labels
    id_scores = 1.0 - scores

    aupr_in = float(
        average_precision_score(
            id_labels,
            id_scores
        )
    )


    # --------------------------------------------------------
    # FPR95
    # --------------------------------------------------------

    fpr95 = compute_fpr95(
        id_uncertainty,
        ood_uncertainty
    )


    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
    }


# ============================================================
# 18. RESULT SCHEMA
# ============================================================

RESULT_COLUMNS = [

    "backbone",
    "framework",

    "id_samples",
    "id_accuracy",
    "id_ece",
    "mean_id_uncertainty",

    "near_ood_samples",
    "mean_near_ood_uncertainty",
    "near_auroc",
    "near_fpr95",
    "near_aupr_in",
    "near_aupr_out",

    "far_ood_samples",
    "mean_far_ood_uncertainty",
    "far_auroc",
    "far_fpr95",
    "far_aupr_in",
    "far_aupr_out",

    "delta_uncertainty_near_minus_id",
    "delta_uncertainty_far_minus_id",

    "checkpoint",
]


# ============================================================
# 19. LOAD EXISTING RESULTS
# ============================================================

if os.path.exists(CSV_PATH):

    results_df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Existing Stage 4 results found: "
        f"{len(results_df)} row(s)"
    )

else:

    results_df = pd.DataFrame(
        columns=RESULT_COLUMNS
    )

    print(
        "No existing Stage 4 results found."
    )


# Ensure expected columns exist
for col in RESULT_COLUMNS:

    if col not in results_df.columns:
        results_df[col] = np.nan

results_df = results_df[
    RESULT_COLUMNS
]


# ============================================================
# 20. COMPLETED CONFIGURATIONS
# ============================================================

completed = set()

for _, row in results_df.iterrows():

    completed.add(
        (
            row["backbone"],
            row["framework"]
        )
    )

print(
    f"Completed configurations: "
    f"{len(completed)}/12"
)

print()


# ============================================================
# 21. ATOMIC CSV SAVE
# ============================================================

def save_results_atomic(df):

    df = df[
        RESULT_COLUMNS
    ].copy()

    fd, temp_path = tempfile.mkstemp(
        suffix=".csv",
        dir=RESULT_DIR
    )

    os.close(fd)

    try:

        df.to_csv(
            temp_path,
            index=False
        )

        os.replace(
            temp_path,
            CSV_PATH
        )

    finally:

        if os.path.exists(temp_path):

            os.remove(
                temp_path
            )


# ============================================================
# 22. STATUS SAVE
# ============================================================

def save_status():

    completed_list = []

    for _, row in results_df.iterrows():

        completed_list.append({
            "backbone": row["backbone"],
            "framework": row["framework"],
        })

    status = {
        "stage": 4,
        "total_configurations": 12,
        "completed_configurations": len(
            completed_list
        ),
        "completed": completed_list,
        "last_update": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    with open(
        STATUS_PATH,
        "w"
    ) as f:

        json.dump(
            status,
            f,
            indent=2
        )


# ============================================================
# 23. SAVE RAW SCORES
# ============================================================

def save_raw_scores(
    backbone,
    framework,
    id_uncertainty,
    near_uncertainty,
    far_uncertainty
):

    filename = (
        f"{backbone}_{framework}_seed42.npz"
    )

    path = os.path.join(
        RAW_SCORE_DIR,
        filename
    )

    np.savez_compressed(
        path,
        id_uncertainty=id_uncertainty,
        near_uncertainty=near_uncertainty,
        far_uncertainty=far_uncertainty,
    )

    return path


# ============================================================
# 24. MAIN 12-CONFIGURATION LOOP
# ============================================================

total_configs = (
    len(BACKBONES)
    * len(FRAMEWORKS)
)

config_number = 0


for backbone in BACKBONES:

    for framework in FRAMEWORKS:

        config_number += 1

        key = (
            backbone,
            framework
        )


        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if key in completed:

            print("=" * 70)
            print(
                f"[{config_number}/{total_configs}] "
                f"SKIPPING COMPLETED: "
                f"{backbone} + {framework}"
            )
            print("=" * 70)
            continue


        print("\n")
        print("=" * 70)
        print(
            f"[{config_number}/{total_configs}] "
            f"{backbone.upper()} + {framework.upper()}"
        )
        print("=" * 70)


        checkpoint_filename = (
            CHECKPOINT_LOOKUP[
                (backbone, framework)
            ]
        )

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            checkpoint_filename
        )

        print(
            "Checkpoint:",
            checkpoint_filename
        )


        model = None

        try:

            # ------------------------------------------------
            # Build model
            # ------------------------------------------------

            model = build_model(
                backbone,
                framework
            )

            # ------------------------------------------------
            # Load checkpoint
            # ------------------------------------------------

            model = load_checkpoint(
                model,
                checkpoint_path
            )

            model.eval()

            print("✓ Model loaded")


            # ------------------------------------------------
            # ID evaluation
            # ------------------------------------------------

            print("\nEvaluating HAM10000 ID...")

            id_result = evaluate_id(
                model,
                framework,
                val_loader
            )

            id_uncertainty = (
                id_result["uncertainties"]
            )

            print(
                f"ID Accuracy : "
                f"{id_result['accuracy']:.4f}"
            )

            print(
                f"ID ECE      : "
                f"{id_result['ece']:.4f}"
            )

            print(
                f"Mean ID U   : "
                f"{id_result['mean_uncertainty']:.6f}"
            )


            # ------------------------------------------------
            # Near-OOD
            # ------------------------------------------------

            print(
                "\nEvaluating PAD-UFES-20 "
                "Near-OOD..."
            )

            near_uncertainty = (
                collect_ood_uncertainty(
                    model,
                    framework,
                    pad_loader
                )
            )

            near_metrics = compute_ood_metrics(
                id_uncertainty,
                near_uncertainty
            )

            mean_near_uncertainty = float(
                np.mean(
                    near_uncertainty
                )
            )

            print(
                f"Near Mean U : "
                f"{mean_near_uncertainty:.6f}"
            )

            print(
                f"Near AUROC  : "
                f"{near_metrics['auroc']:.4f}"
            )

            print(
                f"Near FPR95  : "
                f"{near_metrics['fpr95']:.4f}"
            )

            print(
                f"Near AUPR-In: "
                f"{near_metrics['aupr_in']:.4f}"
            )

            print(
                f"Near AUPR-Out: "
                f"{near_metrics['aupr_out']:.4f}"
            )


            # ------------------------------------------------
            # Far-OOD
            # ------------------------------------------------

            print(
                "\nEvaluating CIFAR-100 "
                "Far-OOD..."
            )

            far_uncertainty = (
                collect_ood_uncertainty(
                    model,
                    framework,
                    cifar_loader
                )
            )

            far_metrics = compute_ood_metrics(
                id_uncertainty,
                far_uncertainty
            )

            mean_far_uncertainty = float(
                np.mean(
                    far_uncertainty
                )
            )

            print(
                f"Far Mean U  : "
                f"{mean_far_uncertainty:.6f}"
            )

            print(
                f"Far AUROC   : "
                f"{far_metrics['auroc']:.4f}"
            )

            print(
                f"Far FPR95   : "
                f"{far_metrics['fpr95']:.4f}"
            )

            print(
                f"Far AUPR-In : "
                f"{far_metrics['aupr_in']:.4f}"
            )

            print(
                f"Far AUPR-Out: "
                f"{far_metrics['aupr_out']:.4f}"
            )


            # ------------------------------------------------
            # Uncertainty gaps
            # ------------------------------------------------

            delta_near = (
                mean_near_uncertainty
                -
                id_result["mean_uncertainty"]
            )

            delta_far = (
                mean_far_uncertainty
                -
                id_result["mean_uncertainty"]
            )


            # ------------------------------------------------
            # Save raw scores
            # ------------------------------------------------

            raw_path = save_raw_scores(
                backbone,
                framework,
                id_uncertainty,
                near_uncertainty,
                far_uncertainty
            )

            print(
                "\n✓ Raw scores saved:",
                raw_path
            )


            # ------------------------------------------------
            # Build result row
            # ------------------------------------------------

            result_row = {

                "backbone":
                    backbone,

                "framework":
                    framework,

                "id_samples":
                    len(id_uncertainty),

                "id_accuracy":
                    id_result["accuracy"],

                "id_ece":
                    id_result["ece"],

                "mean_id_uncertainty":
                    id_result["mean_uncertainty"],

                "near_ood_samples":
                    len(near_uncertainty),

                "mean_near_ood_uncertainty":
                    mean_near_uncertainty,

                "near_auroc":
                    near_metrics["auroc"],

                "near_fpr95":
                    near_metrics["fpr95"],

                "near_aupr_in":
                    near_metrics["aupr_in"],

                "near_aupr_out":
                    near_metrics["aupr_out"],

                "far_ood_samples":
                    len(far_uncertainty),

                "mean_far_ood_uncertainty":
                    mean_far_uncertainty,

                "far_auroc":
                    far_metrics["auroc"],

                "far_fpr95":
                    far_metrics["fpr95"],

                "far_aupr_in":
                    far_metrics["aupr_in"],

                "far_aupr_out":
                    far_metrics["aupr_out"],

                "delta_uncertainty_near_minus_id":
                    delta_near,

                "delta_uncertainty_far_minus_id":
                    delta_far,

                "checkpoint":
                    checkpoint_path,
            }


            # ------------------------------------------------
            # Append result
            # ------------------------------------------------

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [result_row]
                    )
                ],
                ignore_index=True
            )

            results_df = results_df[
                RESULT_COLUMNS
            ]


            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            save_results_atomic(
                results_df
            )

            completed.add(
                key
            )

            save_status()

            print(
                "\n"
                + "=" * 70
            )

            print(
                "✓ CONFIGURATION COMPLETED"
            )

            print(
                f"Saved {len(results_df)}/12 "
                "configurations"
            )

            print(
                "CSV:",
                CSV_PATH
            )

            print(
                "=" * 70
            )


        except Exception as e:

            print("\n")
            print("=" * 70)
            print(
                "ERROR in configuration:"
            )
            print(
                backbone,
                framework
            )
            print("=" * 70)

            print(
                type(e).__name__,
                ":",
                str(e)
            )

            print(
                "\nPreviously completed "
                "configurations remain safely "
                "saved."
            )

            # Free model memory
            if model is not None:

                del model

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            # Stop here rather than silently
            # continuing with a potentially
            # problematic environment.
            raise


        finally:

            if model is not None:

                del model

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


# ============================================================
# 25. FINAL STAGE 4 SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("STAGE 4 COMPLETE")
print("=" * 70)

if os.path.exists(CSV_PATH):

    final_df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Configurations completed: "
        f"{len(final_df)}/12"
    )

    print(
        "\nResults saved to:"
    )

    print(CSV_PATH)

    print(
        "\nFinal result matrix:"
    )

    display(
        final_df[
            [
                "backbone",
                "framework",
                "id_accuracy",
                "id_ece",
                "mean_id_uncertainty",
                "near_auroc",
                "near_fpr95",
                "near_aupr_in",
                "near_aupr_out",
                "far_auroc",
                "far_fpr95",
                "far_aupr_in",
                "far_aupr_out",
            ]
        ].sort_values(
            ["backbone", "framework"]
        ).reset_index(drop=True)
    )

else:

    print(
        "WARNING: Stage 4 CSV was not found."
    )

print("=" * 70)