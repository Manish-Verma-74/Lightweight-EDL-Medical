import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/drive/MyDrive/Lightweight-EDL-Medical/results/stage5_corruption"
SUMMARY_CSV = os.path.join(RESULTS_DIR, "stage5_summary.csv")
RAW_CSV = os.path.join(RESULTS_DIR, "stage5_corruption_results.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "derived_robustness_summary.png")

# Strict File Existence Checks
for path in [SUMMARY_CSV, RAW_CSV]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# 1. Load data
df_summary = pd.read_csv(SUMMARY_CSV)
df_raw = pd.read_csv(RAW_CSV)

# Calculate Gap (Confidence - Accuracy) exclusively for Severity 5
df_s5 = df_raw[df_raw['severity'] == 5].copy()
df_s5['gap'] = (df_s5['mean_confidence'] * 100) - df_s5['accuracy']

# 2. Plotting configurations - MATCHED TO CSV OUTPUT
corruptions = ['gaussian_noise', 'gaussian_blur', 'brightness', 'contrast']
x_labels = ['Noise (S5)', 'Blur (S5)', 'Brightness (S5)', 'Contrast (S5)']
models = ['Softmax + CutMix', 'F-EDL + Standard', 'F-EDL + CutMix']
colors = ['#4C72B0', '#DD8452', '#55A868']

x = np.arange(len(corruptions))
width = 0.25

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.rcParams.update({'font.size': 11})

def get_metric_list(df, model_name, metric_col, corruptions_list):
    model_df = df[df['model'] == model_name]
    values = []
    
    for c in corruptions_list:
        val = model_df.loc[model_df['corruption'] == c, metric_col].values
        if len(val) != 1:
            raise ValueError(
                f"Expected exactly 1 value for model='{model_name}', "
                f"corruption='{c}', metric='{metric_col}', found {len(val)}"
            )
        values.append(float(val[0]))
        
    return values

# Plot 1: Accuracy Drop (from stage5_summary)
for i, model in enumerate(models):
    vals = get_metric_list(df_summary, model, 'accuracy_drop_pp', corruptions)
    axes[0].bar(x + i*width, vals, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

axes[0].set_title('Accuracy Drop: Clean → Severity 5')
axes[0].set_ylabel('Change in Accuracy (Percentage Points)')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
axes[0].legend(loc='lower left', title="Configuration")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: ECE Increase (Computed safely from raw data)
for i, model in enumerate(models):
    model_raw = df_raw[df_raw['model'] == model]
    
    # Get Clean ECE (Severity 0)
    clean_ece_series = model_raw[model_raw['severity'] == 0]['ece'].values
    if len(clean_ece_series) == 0:
        raise ValueError(f"Could not find severity=0 (clean) row for {model}")
    clean_ece = float(clean_ece_series[0])
    
    vals = []
    for c in corruptions:
        # Get Severity 5 ECE
        s5_ece_series = model_raw[(model_raw['corruption'] == c) & (model_raw['severity'] == 5)]['ece'].values
        if len(s5_ece_series) != 1:
            raise ValueError(f"Missing exactly 1 severity=5 row for {model}, {c}")
        s5_ece = float(s5_ece_series[0])
        
        vals.append(s5_ece - clean_ece)
        
    axes[1].bar(x + i*width, vals, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

axes[1].set_title('ECE Increase: Clean → Severity 5')
axes[1].set_ylabel('Increase in Expected Calibration Error')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 3: Confidence-Accuracy Gap (from raw data, severity 5)
for i, model in enumerate(models):
    vals = get_metric_list(df_s5, model, 'gap', corruptions)
    axes[2].bar(x + i*width, vals, width, label=model, color=colors[i], edgecolor='black', linewidth=0.5)

axes[2].set_title('Confidence–Accuracy Gap at Severity 5')
axes[2].set_ylabel('Gap (Percentage Points)')
axes[2].set_xticks(x + width)
axes[2].set_xticklabels(x_labels, rotation=45, ha="right")
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Derived robustness summary saved to {OUTPUT_PLOT}")