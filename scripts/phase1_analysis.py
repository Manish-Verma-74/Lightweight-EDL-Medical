# ============================================================
# PHASE 1 ANALYSIS
# Existing-data analysis only
#
# Outputs:
#   1. Risk-Coverage curves
#   2. Accuracy-Coverage curves
#   3. OOD ROC curves
#   4. ID vs Near-OOD combined distributions
#   5. ID vs Far-OOD combined distributions
#   6. Master and individual Summary CSVs
#
# Uses ONLY existing experiment artifacts.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

ROOT = "/content/drive/MyDrive/Lightweight-EDL-Medical"

ADV_DIR = os.path.join(ROOT, "results", "advanced_metrics")
PRED_DIR = os.path.join(ADV_DIR, "raw_predictions")

OOD_DIR = os.path.join(ROOT, "results", "ood_stage4")
RAW_OOD_DIR = os.path.join(OOD_DIR, "raw_scores")

OUT_DIR = os.path.join(ROOT, "results", "phase1_visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

print("Output directory:")
print(OUT_DIR)

# ------------------------------------------------------------
# 2. MODEL DEFINITIONS
# ------------------------------------------------------------

risk_models = {
    "F-EDL + Standard": (
        os.path.join(PRED_DIR, "ham10000_ham10000_efficientnet_b0_fedl_standard_seed42_predictions.csv"),
        "total_uncertainty"
    ),
    "F-EDL + CutMix": (
        os.path.join(PRED_DIR, "ham10000_ham10000_efficientnet_b0_fedl_cutmix_seed42_predictions.csv"),
        "total_uncertainty"
    ),
    "Softmax + CutMix": (
        os.path.join(PRED_DIR, "ham10000_ham10000_efficientnet_b0_softmax_cutmix_seed42_predictions.csv"),
        "rejection_score"
    ),
}

ood_models = {
    "EfficientNet-B0 + Softmax": os.path.join(RAW_OOD_DIR, "efficientnet_b0_softmax_seed42.npz"),
    "EfficientNet-B0 + EDL": os.path.join(RAW_OOD_DIR, "efficientnet_b0_edl_seed42.npz"),
    "EfficientNet-B0 + R-EDL": os.path.join(RAW_OOD_DIR, "efficientnet_b0_redl_seed42.npz"),
    "EfficientNet-B0 + F-EDL": os.path.join(RAW_OOD_DIR, "efficientnet_b0_fedl_seed42.npz"),
    "MobileNetV3-Small + Softmax": os.path.join(RAW_OOD_DIR, "mobilenet_v3_small_softmax_seed42.npz"),
    "MobileNetV3-Small + EDL": os.path.join(RAW_OOD_DIR, "mobilenet_v3_small_edl_seed42.npz"),
    "MobileNetV3-Small + R-EDL": os.path.join(RAW_OOD_DIR, "mobilenet_v3_small_redl_seed42.npz"),
    "MobileNetV3-Small + F-EDL": os.path.join(RAW_OOD_DIR, "mobilenet_v3_small_fedl_seed42.npz"),
    "ShuffleNetV2 + Softmax": os.path.join(RAW_OOD_DIR, "shufflenet_v2_softmax_seed42.npz"),
    "ShuffleNetV2 + EDL": os.path.join(RAW_OOD_DIR, "shufflenet_v2_edl_seed42.npz"),
    "ShuffleNetV2 + R-EDL": os.path.join(RAW_OOD_DIR, "shufflenet_v2_redl_seed42.npz"),
    "ShuffleNetV2 + F-EDL": os.path.join(RAW_OOD_DIR, "shufflenet_v2_fedl_seed42.npz"),
}


# ------------------------------------------------------------
# 3. SAFETY CHECKS
# ------------------------------------------------------------

print("\nChecking risk-coverage prediction files...")
for name, (path, score_col) in risk_models.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prediction file for {name}:\n{path}")
    
    df = pd.read_csv(path)
    required = ["true_label", "predicted_label", "correct", score_col]
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{name}: prediction file is empty.")
    if df[score_col].isna().any():
        raise ValueError(f"{name}: NaN values in {score_col}.")

    # FIX 1: Integrity check for correct column
    computed_correct = (df["true_label"].to_numpy() == df["predicted_label"].to_numpy()).astype(int)
    stored_correct = df["correct"].to_numpy().astype(int)
    if not np.array_equal(computed_correct, stored_correct):
        raise ValueError(f"{name}: stored 'correct' column does not match true_label == predicted_label.")

    print(f"{name}: {len(df)} samples | score = {score_col}")


print("\nChecking OOD NPZ files...")
for name, path in ood_models.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing OOD file for {name}:\n{path}")
    
    data = np.load(path, allow_pickle=True)
    required = ["id_uncertainty", "near_uncertainty", "far_uncertainty"]
    missing = [k for k in required if k not in data.files]
    
    if missing:
        raise ValueError(f"{name}: missing NPZ keys: {missing}")
        
    id_scores = data['id_uncertainty']
    near_scores = data['near_uncertainty']
    far_scores = data['far_uncertainty']

    # FIX 3: Validation for finite and non-empty OOD scores
    if len(id_scores) == 0 or len(near_scores) == 0 or len(far_scores) == 0:
        raise ValueError(f"{name}: one or more OOD arrays are empty.")
        
    for key, arr in {"id_uncertainty": id_scores, "near_uncertainty": near_scores, "far_uncertainty": far_scores}.items():
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(f"{name}: non-finite values found in {key}.")

    print(f"{name}: ID={len(id_scores)}, Near={len(near_scores)}, Far={len(far_scores)}")


# ------------------------------------------------------------
# 4. RISK-COVERAGE FUNCTION
# ------------------------------------------------------------

def compute_risk_coverage(df, score_col):
    """
    Selective prediction: Lower score = more confident / more suitable to retain.
    """
    scores = df[score_col].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=float)

    order = np.argsort(scores)
    sorted_correct = correct[order]

    n = len(sorted_correct)
    coverage = np.arange(1, n + 1) / n
    cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, n + 1)
    risk = 1.0 - cumulative_accuracy

    return coverage, cumulative_accuracy, risk


# ------------------------------------------------------------
# 5. GENERATE RISK-COVERAGE DATA
# ------------------------------------------------------------

rc_results = {}
rc_summary = []

for name, (path, score_col) in risk_models.items():
    df = pd.read_csv(path)
    coverage, accuracy, risk = compute_risk_coverage(df, score_col)

    rc_results[name] = {
        "coverage": coverage,
        "accuracy": accuracy,
        "risk": risk
    }

    # FIX 2: Do not label np.mean(risk) simply as "AURC"
    mean_empirical_risk = np.mean(risk)

    rc_summary.append({
        "model": name,
        "n_samples": len(df),
        "score_column": score_col,
        "risk_at_25pct": np.interp(0.25, coverage, risk),
        "risk_at_50pct": np.interp(0.50, coverage, risk),
        "risk_at_75pct": np.interp(0.75, coverage, risk),
        "risk_at_100pct": risk[-1],
        "mean_empirical_risk": mean_empirical_risk
    })

rc_summary_df = pd.DataFrame(rc_summary)
rc_summary_path = os.path.join(OUT_DIR, "risk_coverage_summary.csv")
rc_summary_df.to_csv(rc_summary_path, index=False)

print("\nRisk-Coverage summary:")
print(rc_summary_df.to_string(index=False))


# ------------------------------------------------------------
# 6 & 7. RISK-COVERAGE AND ACCURACY-COVERAGE PLOTS
# ------------------------------------------------------------

# Risk-Coverage
plt.figure(figsize=(9, 6))
for name, result in rc_results.items():
    plt.plot(result["coverage"] * 100, result["risk"] * 100, label=name, linewidth=2)
plt.xlabel("Coverage (%)")
plt.ylabel("Risk / Error Rate (%)")
plt.title("Risk-Coverage Curves on HAM10000 Validation Set")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
risk_cov_path = os.path.join(OUT_DIR, "risk_coverage_curves.png")
plt.savefig(risk_cov_path, dpi=300)
plt.close()

# Accuracy-Coverage
plt.figure(figsize=(9, 6))
for name, result in rc_results.items():
    plt.plot(result["coverage"] * 100, result["accuracy"] * 100, label=name, linewidth=2)
plt.xlabel("Coverage (%)")
plt.ylabel("Selective Accuracy (%)")
plt.title("Selective Accuracy vs Coverage on HAM10000 Validation Set")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
acc_cov_path = os.path.join(OUT_DIR, "accuracy_coverage_curves.png")
plt.savefig(acc_cov_path, dpi=300)
plt.close()


# ------------------------------------------------------------
# 8. OOD ROC ANALYSIS
# ------------------------------------------------------------

ood_summary = []
roc_results = {}

for name, path in ood_models.items():
    data = np.load(path, allow_pickle=True)
    id_scores = np.asarray(data["id_uncertainty"], dtype=float)
    near_scores = np.asarray(data["near_uncertainty"], dtype=float)
    far_scores = np.asarray(data["far_uncertainty"], dtype=float)

    # Near-OOD ROC
    y_near = np.concatenate([np.zeros(len(id_scores)), np.ones(len(near_scores))])
    score_near = np.concatenate([id_scores, near_scores])
    near_auc = roc_auc_score(y_near, score_near)
    fpr_near, tpr_near, _ = roc_curve(y_near, score_near)

    # Far-OOD ROC
    y_far = np.concatenate([np.zeros(len(id_scores)), np.ones(len(far_scores))])
    score_far = np.concatenate([id_scores, far_scores])
    far_auc = roc_auc_score(y_far, score_far)
    fpr_far, tpr_far, _ = roc_curve(y_far, score_far)

    roc_results[name] = {
        "near_fpr": fpr_near, "near_tpr": tpr_near, "near_auc": near_auc,
        "far_fpr": fpr_far, "far_tpr": tpr_far, "far_auc": far_auc
    }

    ood_summary.append({
        "model": name,
        "id_samples": len(id_scores),
        "near_ood_samples": len(near_scores),
        "far_ood_samples": len(far_scores),
        "raw_near_AUROC": near_auc,
        "raw_far_AUROC": far_auc,
        "id_mean_uncertainty": np.mean(id_scores),
        "near_mean_uncertainty": np.mean(near_scores),
        "far_mean_uncertainty": np.mean(far_scores),
        "near_delta_uncertainty": np.mean(near_scores) - np.mean(id_scores),
        "far_delta_uncertainty": np.mean(far_scores) - np.mean(id_scores)
    })

ood_summary_df = pd.DataFrame(ood_summary)
ood_summary_path = os.path.join(OUT_DIR, "ood_raw_score_summary.csv")
ood_summary_df.to_csv(ood_summary_path, index=False)


# ------------------------------------------------------------
# 9. CHECK AUROC ORIENTATION AGAINST STAGE-4 CSV
# ------------------------------------------------------------

stage4_csv = os.path.join(OOD_DIR, "stage4_full_ood_results.csv")
if os.path.exists(stage4_csv):
    stage4 = pd.read_csv(stage4_csv)
    print("\nComparing raw-score AUROC with Stage-4 aggregate AUROC...")
    
    comparison_rows = []
    for name, result in roc_results.items():
        parts = name.split(" + ")
        backbone, framework = parts[0], parts[1]

        backbone_map = {
            "EfficientNet-B0": "efficientnet_b0",
            "MobileNetV3-Small": "mobilenet_v3_small",
            "ShuffleNetV2": "shufflenet_v2"
        }
        framework_map = {
            "Softmax": "softmax", "EDL": "edl",
            "R-EDL": "redl", "F-EDL": "fedl"
        }

        b = backbone_map[backbone]
        f = framework_map[framework]
        match = stage4[(stage4["backbone"] == b) & (stage4["framework"] == f)]

        # FIX 5: Exact match check
        if len(match) != 1:
            raise ValueError(f"Expected exactly one Stage-4 row for {name}, found {len(match)}.")

        row = match.iloc[0]
        comparison_rows.append({
            "model": name,
            "raw_near_AUROC": result["near_auc"],
            "stage4_near_AUROC": row["near_auroc"],
            "near_difference": result["near_auc"] - row["near_auroc"],
            "raw_far_AUROC": result["far_auc"],
            "stage4_far_AUROC": row["far_auroc"],
            "far_difference": result["far_auc"] - row["far_auroc"]
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(OUT_DIR, "ood_auroc_validation.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    # FIX 4: Explicit AUROC tolerance validation
    tolerance = 1e-5
    if not comparison_df.empty:
        near_diff = comparison_df["near_difference"].abs().max()
        far_diff = comparison_df["far_difference"].abs().max()

        print(f"\nMaximum near-AUROC difference: {near_diff:.10f}")
        print(f"Maximum far-AUROC difference:  {far_diff:.10f}")

        if near_diff > tolerance or far_diff > tolerance:
            raise ValueError(
                "Raw-score AUROC does not match Stage-4 aggregate AUROC. "
                "Check score orientation or Stage-4 metric implementation."
            )
        else:
            print("AUROC orientation and calculation strictly validated against Stage-4 records.")


# ------------------------------------------------------------
# 10 & 11. NEAR & FAR OOD ROC PLOTS
# ------------------------------------------------------------

# Near-OOD ROC
plt.figure(figsize=(9, 7))
for name, result in roc_results.items():
    plt.plot(result["near_fpr"], result["near_tpr"], linewidth=1.8, label=f"{name} (AUROC={result['near_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Near-OOD ROC Curves")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
near_roc_path = os.path.join(OUT_DIR, "near_ood_roc_curves.png")
plt.savefig(near_roc_path, dpi=300)
plt.close()

# Far-OOD ROC
plt.figure(figsize=(9, 7))
for name, result in roc_results.items():
    plt.plot(result["far_fpr"], result["far_tpr"], linewidth=1.8, label=f"{name} (AUROC={result['far_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Far-OOD ROC Curves")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
far_roc_path = os.path.join(OUT_DIR, "far_ood_roc_curves.png")
plt.savefig(far_roc_path, dpi=300)
plt.close()


# ------------------------------------------------------------
# 12. COMBINED DISTRIBUTION PLOTS (FIX 6 & 7 applied)
# ------------------------------------------------------------

selected_ood = [
    "EfficientNet-B0 + Softmax",
    "EfficientNet-B0 + EDL",
    "EfficientNet-B0 + R-EDL",
    "EfficientNet-B0 + F-EDL",
]

# Near-OOD combined
plt.figure(figsize=(11, 7))
plot_data = []
labels = []

for name in selected_ood:
    data = np.load(ood_models[name], allow_pickle=True)
    plot_data.extend([data["id_uncertainty"], data["near_uncertainty"]])
    labels.extend([f"{name}\nID", f"{name}\nNear-OOD"])

# FIX 7: showfliers=True to reveal long-tail distribution behaviors
plt.boxplot(plot_data, labels=labels, showfliers=True)
plt.ylabel("Uncertainty Score")
plt.title("ID vs Near-OOD Uncertainty — EfficientNet-B0")
plt.xticks(rotation=25, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
combined_near_path = os.path.join(OUT_DIR, "efficientnet_id_vs_near_combined.png")
plt.savefig(combined_near_path, dpi=300)
plt.close()


# Far-OOD combined
plt.figure(figsize=(11, 7))
plot_data = []
labels = []

for name in selected_ood:
    data = np.load(ood_models[name], allow_pickle=True)
    plot_data.extend([data["id_uncertainty"], data["far_uncertainty"]])
    labels.extend([f"{name}\nID", f"{name}\nFar-OOD"])

plt.boxplot(plot_data, labels=labels, showfliers=True)
plt.ylabel("Uncertainty Score")
plt.title("ID vs Far-OOD Uncertainty — EfficientNet-B0")
plt.xticks(rotation=25, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
combined_far_path = os.path.join(OUT_DIR, "efficientnet_id_vs_far_combined.png")
plt.savefig(combined_far_path, dpi=300)
plt.close()


# ------------------------------------------------------------
# 13. MASTER PHASE-1 SUMMARY CSV (FIX 10)
# ------------------------------------------------------------

phase1_summary = pd.concat([
    rc_summary_df.assign(analysis="risk_coverage"),
    ood_summary_df.assign(analysis="ood")
], ignore_index=True)

master_summary_path = os.path.join(OUT_DIR, "phase1_master_summary.csv")
phase1_summary.to_csv(master_summary_path, index=False)
print(f"\nSaved master Phase-1 summary: {master_summary_path}")


# ------------------------------------------------------------
# 14. FINAL FILE LIST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 1 ANALYSIS COMPLETE")
print("=" * 70)
for root, dirs, files in os.walk(OUT_DIR):
    for f in sorted(files):
        full = os.path.join(root, f)
        print(full)
print("\nAll outputs saved under:")
print(OUT_DIR)