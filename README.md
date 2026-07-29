# Lightweight-EDL-Medical

Evidential Deep Learning (EDL) for uncertainty-aware medical image classification.
Stage 1 baseline: HAM10000 skin lesion dataset, lightweight backbones (EfficientNet-B0,
MobileNetV3-Small, ShuffleNetV2), evaluated on accuracy + Expected Calibration Error (ECE).

## Repo structure
```
losses/evidential_loss.py   EDL loss (Sensoy et al. 2018), gives evidence -> Dirichlet -> loss
losses/losses_test.py       Unit tests for the loss (11 tests)
metrics/ece.py              ECE computation + reliability diagram plotting
metrics/metrics_test.py     Unit tests for ECE (10 tests)
models/backbone_factory.py  Builds EfficientNet-B0 / MobileNetV3-Small / ShuffleNetV2
datasets/ham10000.py        HAM10000 dataset loader + weighted sampler for class imbalance
train.py                    Full training loop, checkpointing, results logging
results/master_log.csv      Appended to automatically after each training run
```

## Setup
```bash
pip install -r requirements.txt
```

## 1. Verify everything works (no data needed)
```bash
PYTHONPATH=. python losses/losses_test.py     # expect 11/11 passed
PYTHONPATH=. python metrics/metrics_test.py   # expect 10/10 passed
PYTHONPATH=. python models/backbone_factory.py  # verify output shapes across all 3 backbones
```

## 2. Download HAM10000 (via Kaggle API)
```bash
# requires kaggle.json configured, see https://www.kaggle.com/docs/api
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000 --unzip
PYTHONPATH=. python datasets/ham10000.py data/ham10000
# check data/ham10000/sample_grid.png -- images + labels should look correct
```

## 3. Sanity check training (fast, tiny subset)
```bash
PYTHONPATH=. python train.py --backbone efficientnet_b0 --dataset ham10000 \
    --data_root data/ham10000 --epochs 2 --debug_subset 200
```
Confirm: loss decreases, no crash, a checkpoint gets saved.

## 4. Full baseline run
```bash
PYTHONPATH=. python train.py --backbone efficientnet_b0 --dataset ham10000 \
    --data_root data/ham10000 --epochs 50
```
Results (accuracy, ECE, checkpoint path) get appended to `results/master_log.csv`.

## How the EDL loss works
The network outputs raw "evidence" per class instead of softmax logits:
```
evidence = relu(output)
alpha    = evidence + 1          # Dirichlet concentration parameters
S        = sum(alpha)            # total evidence
prob     = alpha / S             # expected class probabilities
uncertainty = K / S              # higher when the model has little evidence
```
The loss combines a prediction-error term, a variance term, and a KL-divergence
term (annealed in gradually) that discourages evidence piling up on wrong classes.
See `losses/evidential_loss.py` for the full implementation with comments.
