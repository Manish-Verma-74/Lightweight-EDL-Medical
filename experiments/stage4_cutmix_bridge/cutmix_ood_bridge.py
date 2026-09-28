# ============================================================
# STAGE 4 CUTMIX BRIDGE — OOD DETECTION FOR AUGMENTED MODELS
# EfficientNet-B0 × 4 Frameworks = 4 CutMix Configurations
#
# Goal:
#   Assess how CutMix changes OOD detection performance under
#   the evaluated EfficientNet-B0 setup, while preserving exact
#   alignment with the archived Stage-4 Standard evaluation.
#
# ID       : HAM10000 validation (lesion-level)
# Near-OOD : PAD-UFES-20
# Far-OOD  : CIFAR-100
#
# Metrics:
#   ID  : Accuracy, ECE, Mean ID Uncertainty
#   OOD : AUROC, FPR95, AUPR-In, AUPR-Out
#   OOD : Mean Near/Far-OOD Uncertainty and ΔU
#
# Safety / reproducibility:
#   - Verifies all 4 CutMix canonical runs are present
#   - Verifies exact EfficientNet-B0 Stage-4 framework coverage
#   - Loads checkpoints strictly
#   - Recomputes CutMix ID Accuracy/ECE
#   - Requires recomputed Accuracy/ECE to match canonical registry
#     within 1e-6
#   - Uses the original Stage-4 uncertainty definitions
#   - Uses the original Stage-4 empirical FPR95 implementation
#   - Merges CutMix results with archived Standard Stage-4 results
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

# =====================================================================
# 1. SETUP & PATHS
# =====================================================================

PROJECT_ROOT = "/content/drive/MyDrive/Lightweight-EDL-Medical"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"OOD CutMix Bridge (Strictly Aligned) | "
    f"Device: {DEVICE}"
)

CANONICAL = os.path.join(
    PROJECT_ROOT,
    "results",
    "canonical_results_frozen.csv"
)

STAGE4_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "ood_stage4",
    "stage4_full_ood_results.csv"
)

OUT_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage4_cutmix_bridge_summary.csv"
)

os.makedirs(
    os.path.dirname(OUT_CSV),
    exist_ok=True
)

# =====================================================================
# 2. IMPORT PROJECT MODULES
# =====================================================================

from models.fedl_wrapper import FEDLWrapper
from models.backbone_factory import get_backbone
from metrics.ece import compute_ece
from datasets.ham10000 import HAM10000Dataset, default_transforms

from torchvision.datasets import CIFAR100
from torchvision import transforms

from PIL import Image

# =====================================================================
# 3. CONSTANTS
# =====================================================================

NUM_CLASSES = 7
REDL_LAMBDA = 0.1

BATCH_SIZE = 64
IMAGE_SIZE = 224

# Original local runtime dataset locations used for Stage 4
HAM_ROOT = "/content/data/ham10000"
PAD_ROOT = "/content/data/pad_ufes20"
CIFAR_ROOT = "/content/data/cifar100"

# =====================================================================
# 4. LOAD DATASETS
# =====================================================================

print("\nLoading datasets from local runtime...")

# -------------------------------------------------------------
# HAM10000 — ID
# -------------------------------------------------------------

ham_transform = default_transforms(train=False)

ham_data = HAM10000Dataset(
    HAM_ROOT,
    transform=ham_transform
)

sgkf = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

_, val_idx = next(
    sgkf.split(
        np.arange(len(ham_data)),
        ham_data.labels,
        groups=ham_data.metadata["lesion_id"].values
    )
)

id_loader = DataLoader(
    Subset(ham_data, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# -------------------------------------------------------------
# PAD-UFES-20 — Near OOD
# -------------------------------------------------------------

class PADUFES20Dataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):

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

        # Keep only metadata entries whose image files exist
        self.metadata = self.metadata[
            self.metadata["img_id"].isin(
                self.image_paths.keys()
            )
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        img_id = row["img_id"]

        image = Image.open(
            self.image_paths[img_id]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Dummy label because PAD-UFES-20 is OOD-only
        return image, 0


ood_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

pad_dataset = PADUFES20Dataset(
    PAD_ROOT,
    transform=ood_transform
)

near_loader = DataLoader(
    pad_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# -------------------------------------------------------------
# CIFAR-100 — Far OOD
# -------------------------------------------------------------

cifar_dataset = CIFAR100(
    root=CIFAR_ROOT,
    train=False,
    download=False,
    transform=ood_transform
)

far_loader = DataLoader(
    cifar_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =====================================================================
# 5. DATASET SANITY CHECK
# =====================================================================

assert len(val_idx) == 1441, (
    f"Expected 1441 HAM10000 validation samples, "
    f"found {len(val_idx)}"
)

assert len(pad_dataset) == 2298, (
    f"Expected 2298 PAD-UFES-20 samples, "
    f"found {len(pad_dataset)}"
)

assert len(cifar_dataset) == 10000, (
    f"Expected 10000 CIFAR-100 test samples, "
    f"found {len(cifar_dataset)}"
)

print("✓ All datasets loaded successfully")
print(f"  HAM10000 ID    : {len(val_idx)}")
print(f"  PAD-UFES-20    : {len(pad_dataset)}")
print(f"  CIFAR-100      : {len(cifar_dataset)}")

# =====================================================================
# 6. UNCERTAINTY / SCORE EXTRACTION
#    EXACTLY ALIGNED WITH ORIGINAL STAGE 4 DEFINITIONS
# =====================================================================

@torch.no_grad()
def extract_uncertainties(
    model,
    loader,
    framework,
    device
):

    model.eval()

    all_u = []
    all_conf = []
    all_preds = []
    all_labels = []

    for batch in loader:

        # HAM / CIFAR / PAD all provide (image, label)
        if isinstance(batch, (list, tuple)):

            images = batch[0]
            labels = batch[1]

        else:

            images = batch
            labels = torch.zeros(
                len(images),
                dtype=torch.long
            )

        images = images.to(
            device,
            non_blocking=True
        )

        out = model(images)

        # =========================================================
        # SOFTMAX
        # =========================================================

        if framework == "softmax":

            logits = out

            probs = F.softmax(
                logits,
                dim=1
            )

            conf, pred = torch.max(
                probs,
                dim=1
            )

            # Higher uncertainty = more OOD
            u = 1.0 - conf

        # =========================================================
        # STANDARD EDL
        # =========================================================

        elif framework == "edl":

            logits = out

            evidence = F.relu(
                logits
            )

            alpha = evidence + 1.0

            S = torch.sum(
                alpha,
                dim=1
            )

            probs = alpha / S.unsqueeze(1)

            conf, pred = torch.max(
                probs,
                dim=1
            )

            # Original Stage-4 EDL uncertainty
            u = NUM_CLASSES / S

        # =========================================================
        # R-EDL
        # =========================================================

        elif framework == "redl":

            logits = out

            evidence = F.relu(
                logits
            )

            alpha = evidence + REDL_LAMBDA

            S = torch.sum(
                alpha,
                dim=1
            )

            probs = alpha / S.unsqueeze(1)

            conf, pred = torch.max(
                probs,
                dim=1
            )

            # Original Stage-4 R-EDL uncertainty
            u = (
                REDL_LAMBDA
                * NUM_CLASSES
                / S
            )

        # =========================================================
        # F-EDL
        # =========================================================

        elif framework == "fedl":

            # FEDLWrapper returns:
            # alpha, p, tau

            alpha, p, tau = out

            alpha0 = torch.sum(
                alpha,
                dim=1,
                keepdim=True
            )

            mu = (
                alpha + tau * p
            ) / (
                alpha0 + tau
            )

            conf, pred = torch.max(
                mu,
                dim=1
            )

            # IMPORTANT:
            # This is the exact F-EDL uncertainty formulation
            # used in the archived Stage-4 experiment.
            u = (
                1.0
                - torch.sum(
                    mu ** 2,
                    dim=1
                )
            )

        else:

            raise ValueError(
                f"Unknown framework: {framework}"
            )

        all_u.extend(
            u.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_conf.extend(
            conf.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_preds.extend(
            pred.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        if isinstance(labels, torch.Tensor):

            all_labels.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

        else:

            all_labels.extend(
                [0] * len(images)
            )

    return (
        np.asarray(all_u),
        np.asarray(all_conf),
        np.asarray(all_preds),
        np.asarray(all_labels)
    )

# =====================================================================
# 7. FPR95
#    EXACT ORIGINAL STAGE 4 EMPIRICAL IMPLEMENTATION
# =====================================================================

def compute_fpr95(
    id_scores,
    ood_scores
):

    id_scores = np.asarray(
        id_scores
    )

    ood_scores = np.asarray(
        ood_scores
    )

    thresholds = np.sort(
        ood_scores
    )[::-1]

    for threshold in thresholds:

        tpr = np.mean(
            ood_scores >= threshold
        )

        if tpr >= 0.95:

            fpr = np.mean(
                id_scores >= threshold
            )

            return float(fpr)

    return 1.0

# =====================================================================
# 8. OOD METRICS
# =====================================================================

def compute_ood_metrics(
    id_scores,
    ood_scores
):

    id_scores = np.asarray(
        id_scores
    )

    ood_scores = np.asarray(
        ood_scores
    )

    labels = np.concatenate([
        np.zeros(
            len(id_scores)
        ),
        np.ones(
            len(ood_scores)
        )
    ])

    scores = np.concatenate([
        id_scores,
        ood_scores
    ])

    # OOD = positive
    auroc = roc_auc_score(
        labels,
        scores
    )

    aupr_out = average_precision_score(
        labels,
        scores
    )

    # ID = positive
    id_labels = 1 - labels

    id_scores_for_ap = 1 - scores

    aupr_in = average_precision_score(
        id_labels,
        id_scores_for_ap
    )

    fpr95 = compute_fpr95(
        id_scores,
        ood_scores
    )

    return {
        "auroc": float(auroc),
        "fpr95": float(fpr95),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out)
    }

# =====================================================================
# 9. LOAD CANONICAL CUTMIX RUNS
# =====================================================================

df = pd.read_csv(
    CANONICAL
)

bridge_ids = [
    "ham10000_efficientnet_b0_softmax_cutmix_seed42",
    "ham10000_efficientnet_b0_edl_cutmix_seed42",
    "ham10000_efficientnet_b0_redl_cutmix_seed42_lam0.1",
    "ham10000_efficientnet_b0_fedl_cutmix_seed42"
]

bridge = df[
    df["run_id"].isin(bridge_ids)
].copy()

# -------------------------------------------------------------
# SAFETY CHECK 1
# -------------------------------------------------------------

assert len(bridge) == 4, (
    f"Expected 4 CutMix EfficientNet-B0 runs, "
    f"found {len(bridge)}:\n"
    f"{bridge['run_id'].tolist()}"
)

assert set(
    bridge["loss_fn"]
) == {
    "edl",
    "fedl",
    "redl",
    "softmax"
}, (
    "Unexpected framework set in CutMix registry: "
    f"{set(bridge['loss_fn'])}"
)

# =====================================================================
# 10. EVALUATE FOUR CUTMIX MODELS
# =====================================================================

results_list = []

for _, row in bridge.iterrows():

    fw = row["loss_fn"]

    print(
        "\n" + "=" * 65
    )

    print(
        f"Evaluating CutMix EfficientNet-B0 | "
        f"Framework: {fw.upper()}"
    )

    print(
        f"Run ID: {row['run_id']}"
    )

    print(
        "=" * 65
    )

    # -------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------

    if fw == "fedl":

        model = FEDLWrapper(
            "efficientnet_b0",
            num_classes=NUM_CLASSES,
            pretrained=False
        )

    else:

        model = get_backbone(
            "efficientnet_b0",
            num_classes=NUM_CLASSES,
            pretrained=False
        )

    model = model.to(
        DEVICE
    )

    # -------------------------------------------------------------
    # CHECKPOINT
    # -------------------------------------------------------------

    checkpoint_path = row[
        "best_checkpoint_path"
    ]

    assert os.path.exists(
        checkpoint_path
    ), (
        f"Checkpoint not found: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state = checkpoint[
            "state_dict"
        ]

    else:

        state = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state = {
        (
            k[7:]
            if k.startswith("module.")
            else k
        ): v
        for k, v in state.items()
    }

    # Strict loading for exact architecture/checkpoint compatibility
    model.load_state_dict(
        cleaned_state,
        strict=True
    )

    model.eval()

    # -------------------------------------------------------------
    # ID EVALUATION
    # -------------------------------------------------------------

    id_u, id_conf, id_preds, id_labels = (
        extract_uncertainties(
            model,
            id_loader,
            fw,
            DEVICE
        )
    )

    id_acc = float(
        np.mean(
            id_preds == id_labels
        )
    )

    id_ece, _ = compute_ece(
        id_conf,
        id_preds,
        id_labels,
        n_bins=15
    )

    mean_id_u = float(
        np.mean(id_u)
    )

    # -------------------------------------------------------------
    # STRICT CANONICAL REPRODUCTION CHECK
    # -------------------------------------------------------------

    expected_acc = float(
        row["accuracy"]
    )

    expected_ece = float(
        row["ece"]
    )

    if not np.isclose(
        id_acc,
        expected_acc,
        atol=1e-6,
        rtol=0.0
    ):

        raise RuntimeError(
            f"{fw}: ID accuracy mismatch. "
            f"Recomputed={id_acc:.10f}, "
            f"Canonical={expected_acc:.10f}"
        )

    if not np.isclose(
        id_ece,
        expected_ece,
        atol=1e-6,
        rtol=0.0
    ):

        raise RuntimeError(
            f"{fw}: ID ECE mismatch. "
            f"Recomputed={id_ece:.10f}, "
            f"Canonical={expected_ece:.10f}"
        )

    print(
        "✓ ID metrics reproduced within 1e-6 | "
        f"Acc={id_acc:.6f} | "
        f"ECE={id_ece:.6f}"
    )

    # -------------------------------------------------------------
    # NEAR OOD
    # -------------------------------------------------------------

    near_u, _, _, _ = (
        extract_uncertainties(
            model,
            near_loader,
            fw,
            DEVICE
        )
    )

    near_metrics = compute_ood_metrics(
        id_u,
        near_u
    )

    mean_near_u = float(
        np.mean(near_u)
    )

    # -------------------------------------------------------------
    # FAR OOD
    # -------------------------------------------------------------

    far_u, _, _, _ = (
        extract_uncertainties(
            model,
            far_loader,
            fw,
            DEVICE
        )
    )

    far_metrics = compute_ood_metrics(
        id_u,
        far_u
    )

    mean_far_u = float(
        np.mean(far_u)
    )

    # -------------------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------------------

    results_list.append({

        "backbone":
            "efficientnet_b0",

        "framework":
            fw,

        "augmentation":
            "cutmix",

        # ID
        "id_samples":
            len(id_labels),

        "id_accuracy":
            id_acc,

        "id_ece":
            float(id_ece),

        "mean_id_uncertainty":
            mean_id_u,

        # Near OOD
        "near_ood_samples":
            len(near_u),

        "near_auroc":
            near_metrics["auroc"],

        "near_fpr95":
            near_metrics["fpr95"],

        "near_aupr_in":
            near_metrics["aupr_in"],

        "near_aupr_out":
            near_metrics["aupr_out"],

        "mean_near_ood_uncertainty":
            mean_near_u,

        "delta_uncertainty_near_minus_id":
            mean_near_u - mean_id_u,

        # Far OOD
        "far_ood_samples":
            len(far_u),

        "far_auroc":
            far_metrics["auroc"],

        "far_fpr95":
            far_metrics["fpr95"],

        "far_aupr_in":
            far_metrics["aupr_in"],

        "far_aupr_out":
            far_metrics["aupr_out"],

        "mean_far_ood_uncertainty":
            mean_far_u,

        "delta_uncertainty_far_minus_id":
            mean_far_u - mean_id_u
    })

    print(
        f"Near-OOD | "
        f"AUROC={near_metrics['auroc']:.6f} | "
        f"FPR95={near_metrics['fpr95']:.6f} | "
        f"AUPR-In={near_metrics['aupr_in']:.6f} | "
        f"AUPR-Out={near_metrics['aupr_out']:.6f}"
    )

    print(
        f"Far-OOD  | "
        f"AUROC={far_metrics['auroc']:.6f} | "
        f"FPR95={far_metrics['fpr95']:.6f} | "
        f"AUPR-In={far_metrics['aupr_in']:.6f} | "
        f"AUPR-Out={far_metrics['aupr_out']:.6f}"
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

# =====================================================================
# 11. LOAD ARCHIVED STAGE-4 STANDARD RESULTS
# =====================================================================

print(
    "\nLoading archived Stage-4 Standard results..."
)

df_stage4 = pd.read_csv(
    STAGE4_CSV
)

df_stage4_eff = df_stage4[
    df_stage4["backbone"] == "efficientnet_b0"
].copy()

# -------------------------------------------------------------
# SAFETY CHECK 2
# -------------------------------------------------------------

expected_frameworks = {
    "edl",
    "fedl",
    "redl",
    "softmax"
}

assert set(
    df_stage4_eff["framework"]
) == expected_frameworks, (
    f"Expected Stage-4 frameworks "
    f"{expected_frameworks}, "
    f"found {set(df_stage4_eff['framework'])}"
)

assert len(df_stage4_eff) == 4, (
    f"Expected exactly 4 EfficientNet-B0 "
    f"Stage-4 Standard rows, "
    f"found {len(df_stage4_eff)}"
)

df_stage4_eff[
    "augmentation"
] = "standard"

# =====================================================================
# 12. ALIGN STANDARD + CUTMIX STRUCTURE
# =====================================================================

df_cutmix = pd.DataFrame(
    results_list
)

# The archived Stage-4 file is expected to contain the same
# metric columns used by the bridge.

all_cols = (
    df_cutmix.columns.tolist()
)

missing_stage4_cols = [
    c for c in all_cols
    if c not in df_stage4_eff.columns
]

assert not missing_stage4_cols, (
    "Archived Stage-4 CSV is missing required columns: "
    f"{missing_stage4_cols}"
)

df_stage4_eff = df_stage4_eff[
    all_cols
]

# =====================================================================
# 13. COMBINE
# =====================================================================

df_combined = pd.concat(
    [
        df_stage4_eff,
        df_cutmix
    ],
    ignore_index=True
)

df_combined = (
    df_combined
    .sort_values(
        by=[
            "framework",
            "augmentation"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)

# =====================================================================
# 14. SAVE
# =====================================================================

df_combined.to_csv(
    OUT_CSV,
    index=False
)

print(
    "\n✅ Stage-4 CutMix OOD bridge completed."
)

print(
    f"Saved to:\n{OUT_CSV}"
)

print(
    f"\nTotal rows: {len(df_combined)} "
    f"(4 Standard + 4 CutMix)"
)

# =====================================================================
# 15. SUMMARY TABLES
# =====================================================================

print(
    "\n" + "=" * 70
)

print(
    "STANDARD vs CUTMIX — EfficientNet-B0"
)

print(
    "=" * 70
)

# -------------------------------------------------------------
# ECE
# -------------------------------------------------------------

pivot_ece = df_combined.pivot(
    index="framework",
    columns="augmentation",
    values="id_ece"
)

print(
    "\n1. ID Calibration (ECE)"
)

print(
    pivot_ece
)

# -------------------------------------------------------------
# Near-OOD AUROC
# -------------------------------------------------------------

pivot_near_auroc = df_combined.pivot(
    index="framework",
    columns="augmentation",
    values="near_auroc"
)

print(
    "\n2. Near-OOD Detection (AUROC)"
)

print(
    pivot_near_auroc
)

# -------------------------------------------------------------
# Far-OOD AUROC
# -------------------------------------------------------------

pivot_far_auroc = df_combined.pivot(
    index="framework",
    columns="augmentation",
    values="far_auroc"
)

print(
    "\n3. Far-OOD Detection (AUROC)"
)

print(
    pivot_far_auroc
)

# -------------------------------------------------------------
# Near-OOD FPR95
# -------------------------------------------------------------

pivot_near_fpr95 = df_combined.pivot(
    index="framework",
    columns="augmentation",
    values="near_fpr95"
)

print(
    "\n4. Near-OOD FPR95"
)

print(
    pivot_near_fpr95
)

# -------------------------------------------------------------
# Far-OOD FPR95
# -------------------------------------------------------------

pivot_far_fpr95 = df_combined.pivot(
    index="framework",
    columns="augmentation",
    values="far_fpr95"
)

print(
    "\n5. Far-OOD FPR95"
)

print(
    pivot_far_fpr95
)

# =====================================================================
# 16. FINAL CHECK
# =====================================================================

assert len(df_combined) == 8

assert set(
    df_combined["augmentation"]
) == {
    "standard",
    "cutmix"
}

assert set(
    df_combined["framework"]
) == {
    "edl",
    "fedl",
    "redl",
    "softmax"
}

print(
    "\n✅ Final structural validation passed."
)