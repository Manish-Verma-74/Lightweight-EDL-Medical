# ============================================================
# STAGE 4 CUTMIX BRIDGE — OOD DETECTION FOR AUGMENTED MODELS
# 1 Backbone (EfficientNet-B0) × 4 Frameworks = 4 Configurations
#
# Goal: Bridges Stage 3 (CutMix Interaction) with Stage 4 (OOD) 
#       to determine if CutMix degrades OOD detection capability.
#
# ID      : HAM10000 validation (lesion-level)
# Near-OOD: PAD-UFES-20
# Far-OOD : CIFAR-100
#
# Metrics:
#   ID  : Accuracy, ECE, Mean ID Uncertainty
#   OOD : AUROC, FPR95, AUPR-In, AUPR-Out
#   OOD : Mean Near/Far-OOD Uncertainty & ΔU
#
# Restart-safe:
#   - Asserts exact reproduction of canonical ID metrics (Acc/ECE)
#   - Merges output seamlessly with existing Stage 4 Standard results
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# =====================================================================
# 1. SETUP & PATHS
# =====================================================================
PROJECT_ROOT = "/content/drive/MyDrive/Lightweight-EDL-Medical"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"OOD CutMix Bridge (4 Models) | Device: {DEVICE}")

CANONICAL = os.path.join(PROJECT_ROOT, "results", "canonical_results_frozen.csv")
STAGE4_CSV = os.path.join(PROJECT_ROOT, "results", "ood_stage4", "stage4_full_ood_results.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "results", "stage4_cutmix_bridge_summary.csv")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# Import Stage 4 modules
from models.fedl_wrapper import FEDLWrapper
from models.backbone_factory import get_backbone
from metrics.ece import compute_ece
from datasets.ham10000 import HAM10000Dataset, default_transforms
from torchvision.datasets import CIFAR100
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import PIL.Image

NUM_CLASSES = 7
REDL_LAMBDA = 0.1
BATCH_SIZE = 64
IMAGE_SIZE = 224

# =====================================================================
# 2. LOAD DATASETS
# =====================================================================
print("\nLoading Datasets...")
transform = default_transforms(train=False)

# ID
ham_data = HAM10000Dataset("/content/data/ham10000", transform=transform)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
_, val_idx = next(sgkf.split(np.arange(len(ham_data)), ham_data.labels, groups=ham_data.metadata["lesion_id"].values))
id_loader = DataLoader(Subset(ham_data, val_idx), batch_size=BATCH_SIZE, shuffle=False)

# Near-OOD
class PADUFES20Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        self.image_paths = {}
        images_root = os.path.join(root_dir, "images")
        for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
            folder = os.path.join(images_root, part)
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths[fname] = os.path.join(folder, fname)
        self.metadata = self.metadata[self.metadata["img_id"].isin(self.image_paths.keys())].reset_index(drop=True)

    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = PIL.Image.open(self.image_paths[row["img_id"]]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, 0 

ood_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pad_dataset = PADUFES20Dataset("/content/data/pad_ufes20", transform=ood_transform)
near_loader = DataLoader(pad_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Far-OOD
cifar_dataset = CIFAR100(root="/content/data/cifar100", train=False, download=False, transform=ood_transform)
far_loader = DataLoader(cifar_dataset, batch_size=BATCH_SIZE, shuffle=False)

assert len(val_idx) == 1441 and len(pad_dataset) == 2298 and len(cifar_dataset) == 10000

# =====================================================================
# 3. METRIC FUNCTIONS
# =====================================================================
def extract_uncertainties(model, loader, framework, device):
    import torch.nn.functional as F
    model.eval()
    all_u, all_conf, all_preds, all_labels = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            if framework == "softmax":
                prob = F.softmax(out, dim=1)
                conf, pred = torch.max(prob, dim=1)
                u = 1.0 - conf
            elif framework == "edl":
                alpha = F.relu(out) + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                conf, pred = torch.max(alpha / S, dim=1)
                u = (NUM_CLASSES / S).squeeze(1)
            elif framework == "redl":
                alpha = F.relu(out) + REDL_LAMBDA
                S = torch.sum(alpha, dim=1, keepdim=True)
                conf, pred = torch.max(alpha / S, dim=1)
                u = (REDL_LAMBDA * NUM_CLASSES / S).squeeze(1)
            elif framework == "fedl":
                alpha, p, tau = out
                S = alpha.sum(dim=1, keepdim=True) + tau
                mu = (alpha + tau * p) / S
                conf, pred = torch.max(mu, dim=1)
                var = (mu * (1.0 - mu) / (S + 1.0)) + ((tau**2) * p * (1.0 - p) / (S * (S + 1.0)))
                u = var.sum(dim=1)
            all_u.extend(u.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else [0]*len(images))
    return np.array(all_u), np.array(all_conf), np.array(all_preds), np.array(all_labels)

def compute_fpr95(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= 0.95)[0]
    return float(fpr[valid[0]]) if len(valid) > 0 else float("nan")

def compute_ood_metrics(id_u, ood_u):
    labels = np.concatenate([np.zeros(len(id_u)), np.ones(len(ood_u))])
    scores = np.concatenate([id_u, ood_u])
    auroc = float(roc_auc_score(labels, scores))
    aupr_out = float(average_precision_score(labels, scores))
    aupr_in = float(average_precision_score(1.0 - labels, 1.0 - scores))
    fpr95 = compute_fpr95(id_u, ood_u)
    return {"auroc": auroc, "fpr95": fpr95, "aupr_in": aupr_in, "aupr_out": aupr_out}

# =====================================================================
# 4. EXECUTION LOOP
# =====================================================================
df = pd.read_csv(CANONICAL)
bridge_ids = [
    "ham10000_efficientnet_b0_softmax_cutmix_seed42",
    "ham10000_efficientnet_b0_edl_cutmix_seed42",
    "ham10000_efficientnet_b0_redl_cutmix_seed42_lam0.1",
    "ham10000_efficientnet_b0_fedl_cutmix_seed42",
]
bridge = df[df["run_id"].isin(bridge_ids)].copy()

# SAFETY CHECK 1: Ensure all 4 CutMix runs are loaded
assert len(bridge) == 4, (
    f"Expected 4 CutMix EfficientNet-B0 runs, found {len(bridge)}: "
    f"{bridge['run_id'].tolist()}"
)

results_list = []

for _, row in bridge.iterrows():
    fw = row['loss_fn']
    print(f"\nEvaluating CutMix Model: {fw.upper()}")
    
    # 1. Build Model
    if fw == 'fedl':
        model = FEDLWrapper('efficientnet_b0', num_classes=7, pretrained=False).to(DEVICE)
    else:
        model = get_backbone('efficientnet_b0', num_classes=7, pretrained=False).to(DEVICE)
    
    # 2. Strict Checkpoint Loading
    checkpoint = torch.load(row['best_checkpoint_path'], map_location=DEVICE)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {fw}: missing={missing}, unexpected={unexpected}")
    
    # 3. ID Extract & STRICT SANITY CHECK
    id_u, id_conf, id_preds, id_labels = extract_uncertainties(model, id_loader, fw, DEVICE)
    id_acc = float(np.mean(id_preds == id_labels))
    id_ece, _ = compute_ece(id_conf, id_preds, id_labels, n_bins=15)
    mean_id_u = float(np.mean(id_u))
    
    expected_acc = float(row["accuracy"])
    expected_ece = float(row["ece"])

    if not np.isclose(id_acc, expected_acc, atol=1e-6):
        raise RuntimeError(f"{fw}: ID accuracy mismatch. Recomputed={id_acc:.6f}, Canonical={expected_acc:.6f}")
    if not np.isclose(id_ece, expected_ece, atol=1e-6):
        raise RuntimeError(f"{fw}: ID ECE mismatch. Recomputed={id_ece:.6f}, Canonical={expected_ece:.6f}")

    print(f"  ✓ ID metrics exactly reproduced | Acc={id_acc:.4f} | ECE={id_ece:.4f}")
    
    # 4. Near & Far Extract
    near_u, _, _, _ = extract_uncertainties(model, near_loader, fw, DEVICE)
    near_metrics = compute_ood_metrics(id_u, near_u)
    mean_near_u = float(np.mean(near_u))
    
    far_u, _, _, _ = extract_uncertainties(model, far_loader, fw, DEVICE)
    far_metrics = compute_ood_metrics(id_u, far_u)
    mean_far_u = float(np.mean(far_u))
    
    # 5. Full Metrics Append 
    results_list.append({
        "backbone": "efficientnet_b0",
        "framework": fw,
        "augmentation": "cutmix",
        "id_samples": len(id_labels),
        "id_accuracy": id_acc,
        "id_ece": id_ece,
        "mean_id_uncertainty": mean_id_u,
        
        "near_ood_samples": len(near_u),
        "near_auroc": near_metrics["auroc"],
        "near_fpr95": near_metrics["fpr95"],
        "near_aupr_in": near_metrics["aupr_in"],
        "near_aupr_out": near_metrics["aupr_out"],
        "mean_near_ood_uncertainty": mean_near_u,
        "delta_uncertainty_near_minus_id": mean_near_u - mean_id_u,
        
        "far_ood_samples": len(far_u),
        "far_auroc": far_metrics["auroc"],
        "far_fpr95": far_metrics["fpr95"],
        "far_aupr_in": far_metrics["aupr_in"],
        "far_aupr_out": far_metrics["aupr_out"],
        "mean_far_ood_uncertainty": mean_far_u,
        "delta_uncertainty_far_minus_id": mean_far_u - mean_id_u,
    })

# =====================================================================
# 5. MERGE & PRINT SUMMARY
# =====================================================================
print("\nLoading existing Stage 4 results to merge...")
df_stage4 = pd.read_csv(STAGE4_CSV)
df_stage4_eff = df_stage4[df_stage4['backbone'] == 'efficientnet_b0'].copy()
df_stage4_eff['augmentation'] = 'standard' 

# SAFETY CHECK 2: Ensure strictly the 4 expected EfficientNet-B0 Stage-4 rows exist
expected_frameworks = {"edl", "fedl", "redl", "softmax"}
assert set(df_stage4_eff["framework"]) == expected_frameworks, (
    f"Expected frameworks {expected_frameworks}, but found {set(df_stage4_eff['framework'])}"
)
assert len(df_stage4_eff) == 4, (
    f"Expected exactly 4 EfficientNet-B0 Stage-4 rows, found {len(df_stage4_eff)}"
)

# Align and merge
df_cutmix = pd.DataFrame(results_list)
all_cols = df_cutmix.columns.tolist()
df_stage4_eff = df_stage4_eff[all_cols] 

df_combined = pd.concat([df_stage4_eff, df_cutmix], ignore_index=True)
df_combined = df_combined.sort_values(by=["framework", "augmentation"], ascending=[True, False]).reset_index(drop=True)
df_combined.to_csv(OUT_CSV, index=False)

print(f"\n✅ 8-Model OOD Bridge completed and saved to:\n  {OUT_CSV}")

print("\n" + "="*55)
print("FINAL BRIDGE VERDICT: STANDARD vs CUTMIX (EfficientNet-B0)")
print("="*55)

pivot_ece = df_combined.pivot(index="framework", columns="augmentation", values="id_ece")
print("\n1. In-Distribution Calibration (ECE) - Lower is better:")
print(pivot_ece)

pivot_near = df_combined.pivot(index="framework", columns="augmentation", values="near_auroc")
print("\n2. Near-OOD Detection (AUROC) - Higher is better:")
print(pivot_near)
