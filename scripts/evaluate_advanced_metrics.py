import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold

# Import existing modules
from datasets.ham10000 import HAM10000Dataset, default_transforms
from models.backbone_factory import get_backbone
from models.fedl_wrapper import FEDLWrapper
from losses.fedl_loss import fedl_predictions
from metrics.ece import compute_ece 

def safe_nanmean(x):
    """Safely computes nanmean to avoid RuntimeWarning on all-NaN slices."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def compute_cw_ece(probs, labels, n_bins=15):
    """Compute one-vs-rest class-wise ECE and Macro-cwECE using np.digitize."""
    N, K = probs.shape
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    class_eces = {}
    macro_cwece = 0.0

    for k in range(K):
        p_k = probs[:, k]
        y_k = (labels == k).astype(float)

        # Assign every probability to exactly one bin, including 0 and 1.
        bin_ids = np.digitize(p_k, bin_boundaries[1:-1], right=True)
        ece_k = 0.0

        for b in range(n_bins):
            in_bin = (bin_ids == b)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(y_k[in_bin])
                conf_in_bin = np.mean(p_k[in_bin])
                ece_k += abs(conf_in_bin - acc_in_bin) * prop_in_bin

        class_eces[f"cwECE_{k}"] = float(ece_k)
        macro_cwece += ece_k

    return float(macro_cwece / K), class_eces

def compute_brier_score(probs, labels):
    """Computes multiclass Brier Score."""
    N, K = probs.shape
    y_onehot = np.zeros((N, K))
    y_onehot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probs - y_onehot)**2, axis=1)))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    repo_root = "/content/drive/MyDrive/Lightweight-EDL-Medical"
    data_root = "/content/data/ham10000"
    frozen_log_path = os.path.join(repo_root, "results", "canonical_results_frozen.csv")
    
    out_dir = os.path.join(repo_root, "results", "advanced_metrics")
    raw_pred_dir = os.path.join(out_dir, "raw_predictions")
    os.makedirs(raw_pred_dir, exist_ok=True)
    summary_csv_path = os.path.join(out_dir, "advanced_metrics_summary.csv")
    
    df = pd.read_csv(frozen_log_path)
    
    # Target 3 shortlisted models matching the corruption study
    shortlist_mask = (
        (df['backbone'] == 'efficientnet_b0') & 
        (
            ((df['loss_fn'] == 'softmax') & (df['augmentation'] == 'cutmix')) |
            ((df['loss_fn'] == 'fedl') & (df['augmentation'] == 'standard')) |
            ((df['loss_fn'] == 'fedl') & (df['augmentation'] == 'cutmix'))
        )
    )
    df_shortlist = df[shortlist_mask].copy()
    
    # Assert exact number of target experiments
    expected_runs = 3
    if len(df_shortlist) != expected_runs:
        raise ValueError(f"Expected {expected_runs} shortlisted experiments, found {len(df_shortlist)}")
        
    print("\nShortlisted experiments:")
    print(df_shortlist[["run_id", "backbone", "loss_fn", "augmentation", "seed"]].to_string(index=False))
    
    results_list = []
    
    for idx, row in df_shortlist.iterrows():
        print(f"\nEvaluating {row['loss_fn'].upper()} + {row['augmentation']} on {row['backbone']}...")
        
        # 1. Exact canonical split reconstruction
        dataset = HAM10000Dataset(data_root, transform=default_transforms(train=False))
        all_indices = np.arange(len(dataset))
        all_labels = np.array(dataset.labels)
        all_lesion_ids = dataset.metadata["lesion_id"].values
        
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=int(row['seed']))
        train_indices, val_indices = next(sgkf.split(all_indices, all_labels, groups=all_lesion_ids))
        
        # Split Defenses
        train_lesions = set(all_lesion_ids[train_indices])
        val_lesions = set(all_lesion_ids[val_indices])
        overlap = train_lesions.intersection(val_lesions)
        
        if overlap:
            raise ValueError(f"Lesion leakage detected! {len(overlap)} lesion IDs overlap.")
        if len(val_indices) != 1441:
            raise ValueError(f"Split reconstruction failed! Expected 1441 validation samples, got {len(val_indices)}")
        if len(np.unique(val_indices)) != len(val_indices):
            raise ValueError("Duplicate validation indices detected.")
            
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=64, shuffle=False)
        
        # 2. Load BEST checkpoint
        ckpt_path = os.path.join(repo_root, "checkpoints", f"{row['run_id']}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        if row['loss_fn'] == 'fedl':
            model = FEDLWrapper(row['backbone'], num_classes=7, pretrained=False).to(device)
        else:
            model = get_backbone(row['backbone'], num_classes=7, pretrained=False).to(device)
            
        print(f"  ↳ Loading best checkpoint: {os.path.basename(ckpt_path)}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        model.eval()
        
        all_probs, all_preds, all_labels_eval, all_conf = [], [], [], []
        all_total_uncert, all_vacuity, all_rejection_score = [], [], []
        
        # 3. Forward Pass
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                out = model(images)
                
                if row['loss_fn'] == 'fedl':
                    alpha, p, tau = out
                    _, conf, total_uncert, _, _ = fedl_predictions(alpha, p, tau)
                    denom = alpha.sum(dim=1, keepdim=True) + tau
                    probs = (alpha + tau * p) / denom
                    
                    tu = total_uncert
                    vac = 7.0 / alpha.sum(dim=1)
                    rej = tu
                    
                elif row['loss_fn'] == 'softmax':
                    probs = torch.softmax(out, dim=1)
                    conf = torch.max(probs, dim=1)[0]
                    
                    tu = torch.full_like(conf, float('nan'))
                    vac = torch.full_like(conf, float('nan'))
                    rej = 1.0 - conf
                
                preds = torch.argmax(probs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels_eval.extend(labels.numpy())
                all_conf.extend(conf.cpu().numpy())
                all_total_uncert.extend(tu.cpu().numpy())
                all_vacuity.extend(vac.cpu().numpy())
                all_rejection_score.extend(rej.cpu().numpy())
                
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels_eval = np.array(all_labels_eval)
        
        # Strict Probability & Array Integrity Defenses
        if not np.all((all_labels_eval >= 0) & (all_labels_eval < 7)):
            raise ValueError("Invalid class labels detected.")
            
        if not (len(all_preds) == len(all_conf) == len(all_total_uncert) == 
                len(all_vacuity) == len(all_rejection_score) == 1441):
            raise ValueError("Per-sample output lengths are inconsistent.")

        if all_probs.shape != (1441, 7):
            raise ValueError(f"Expected probability shape (1441, 7), got {all_probs.shape}")
        if not np.all(np.isfinite(all_probs)):
            raise ValueError("Non-finite probability values detected.")
        if np.any(all_probs < -1e-6) or np.any(all_probs > 1 + 1e-6):
            raise ValueError("Probability outside [0,1] detected.")
            
        row_sums = all_probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(f"Probability rows do not sum to 1. Min={row_sums.min():.8f}, Max={row_sums.max():.8f}")
        
        # 4. Calculate Metrics
        eval_acc = float(np.mean(all_preds == all_labels_eval))
        eval_ece, _ = compute_ece(all_conf, all_preds, all_labels_eval, n_bins=15)
        eval_macro_cwece, class_eces = compute_cw_ece(all_probs, all_labels_eval)
        eval_brier = compute_brier_score(all_probs, all_labels_eval)
        
        # Direct Provenance Check for F-EDL + Standard against Stage 4 OOD table
        if row['loss_fn'] == 'fedl' and row['augmentation'] == 'standard':
            print(f"  ↳ [OOD Cross-Check] ID Accuracy: {eval_acc:.6f} (Expected ≈ 0.850104)")
        
        # 5. Save Per-Sample Predictions
        per_sample_dict = {
            'dataset_index': val_indices,
            'true_label': all_labels_eval,
            'predicted_label': all_preds,
            'correct': (all_preds == all_labels_eval).astype(int),
            'confidence': all_conf,
            'rejection_score': all_rejection_score,
            'total_uncertainty': all_total_uncert,
            'vacuity': all_vacuity
        }
        for k in range(7):
            per_sample_dict[f'prob_class_{k}'] = all_probs[:, k]
            
        per_sample_df = pd.DataFrame(per_sample_dict)
        raw_csv_name = f"ham10000_{row['run_id']}_predictions.csv"
        per_sample_df.to_csv(os.path.join(raw_pred_dir, raw_csv_name), index=False)
        
        # 6. Store Aggregates via Safe NaN Means
        row_dict = row.to_dict()
        row_dict['eval_accuracy_best_ckpt'] = eval_acc
        row_dict['eval_ece_best_ckpt'] = eval_ece
        row_dict['eval_macro_cwece'] = eval_macro_cwece
        row_dict['eval_brier'] = eval_brier
        row_dict['mean_confidence'] = float(np.mean(all_conf))
        row_dict['mean_rejection_score'] = float(np.mean(all_rejection_score))
        row_dict['mean_total_uncertainty'] = safe_nanmean(all_total_uncert)
        row_dict['mean_vacuity'] = safe_nanmean(all_vacuity)
        row_dict.update(class_eces)
        results_list.append(row_dict)
        
        print(f"  ↳ Acc: {eval_acc:.4f} | ECE: {eval_ece:.4f} | Macro-cwECE: {eval_macro_cwece:.4f} | Brier: {eval_brier:.4f}")
        
    out_df = pd.DataFrame(results_list)
    out_df.to_csv(summary_csv_path, index=False)
    print(f"\n✅ Advanced metrics summary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()