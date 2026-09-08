import torch

from losses.evidential_loss import redl_loss, redl_predictions


# Reproducible random numbers
torch.manual_seed(0)

# Fake HAM10000-like batch
fake_output = torch.randn(4, 7)
fake_target = torch.randint(0, 7, (4,))

device = torch.device("cpu")


# -----------------------------
# Test R-EDL loss
# -----------------------------
loss = redl_loss(
    fake_output,
    fake_target,
    epoch_num=1,
    num_classes=7,
    annealing_step=10,
    device=device,
    lam=0.1,
)

print("redl_loss:", loss.item())


# -----------------------------
# Test predictions
# -----------------------------
pred, conf, unc = redl_predictions(
    fake_output,
    lam=0.1
)

print("pred:", pred)
print("confidence:", conf)
print("uncertainty:", unc)


# -----------------------------
# Basic numerical checks
# -----------------------------
assert torch.isfinite(loss), "Loss is NaN or Inf!"
assert loss.item() > 0, "Loss should be positive!"

assert pred.shape == (4,), "Prediction shape incorrect!"
assert conf.shape == (4,), "Confidence shape incorrect!"
assert unc.shape == (4,), "Uncertainty shape incorrect!"

assert torch.all(conf >= 0), "Confidence < 0!"
assert torch.all(conf <= 1), "Confidence > 1!"

assert torch.all(torch.isfinite(unc)), "Uncertainty contains NaN/Inf!"

print("\nR-EDL sanity check PASSED!")