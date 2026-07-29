"""
Run with:  PYTHONPATH=. python losses/losses_test.py
"""
import torch
from losses.evidential_loss import edl_mse_loss, relu_evidence, kl_divergence, edl_predictions

device = torch.device("cpu")
passed = 0
total = 0


def check(name, condition):
    global passed, total
    total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    print(f"[{status}] {name}")


# 1. relu_evidence never negative
x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])
check("relu_evidence_nonnegative", torch.all(relu_evidence(x) >= 0))

# 2. relu_evidence passes positive values through unchanged
check("relu_evidence_identity_on_positive", torch.allclose(relu_evidence(torch.tensor([1.0, 2.5])), torch.tensor([1.0, 2.5])))

# 3. KL divergence of uniform alpha vs itself is ~0
alpha_uniform = torch.ones((1, 4))
kl = kl_divergence(alpha_uniform, 4, device)
check("kl_uniform_vs_uniform_near_zero", torch.allclose(kl, torch.zeros_like(kl), atol=1e-4))

# 4. KL divergence increases as alpha moves away from uniform
alpha_peaked = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
kl_peaked = kl_divergence(alpha_peaked, 4, device)
check("kl_peaked_greater_than_uniform", (kl_peaked > kl).all())

# 5. Loss is finite and non-negative for a random batch
output = torch.randn(8, 5, requires_grad=True)
target = torch.randint(0, 5, (8,))
loss = edl_mse_loss(output, target, epoch_num=1, num_classes=5, annealing_step=10, device=device)
check("loss_is_finite", torch.isfinite(loss))
check("loss_is_nonnegative", loss.item() >= 0)

# 6. Loss supports backward pass (gradients flow)
loss.backward()
check("gradients_flow_to_output", output.grad is not None and torch.any(output.grad != 0))

# 7. Strong correct evidence gives lower loss than no evidence at all
strong_correct = torch.zeros(1, 3)
strong_correct[0, 0] = 20.0  # huge evidence for class 0
target1 = torch.tensor([0])
loss_strong = edl_mse_loss(strong_correct, target1, epoch_num=10, num_classes=3, annealing_step=10, device=device)

no_evidence = torch.zeros(1, 3)
loss_none = edl_mse_loss(no_evidence, target1, epoch_num=10, num_classes=3, annealing_step=10, device=device)
check("strong_correct_evidence_lowers_loss", loss_strong.item() < loss_none.item())

# 8. Strong WRONG evidence gives higher loss than no evidence
strong_wrong = torch.zeros(1, 3)
strong_wrong[0, 1] = 20.0  # huge evidence for class 1, but target is class 0
loss_wrong = edl_mse_loss(strong_wrong, target1, epoch_num=10, num_classes=3, annealing_step=10, device=device)
check("strong_wrong_evidence_raises_loss", loss_wrong.item() > loss_none.item())

# 9. Annealing coefficient behavior: KL term contributes less at epoch 0 than at epoch >= annealing_step
loss_early = edl_mse_loss(strong_wrong, target1, epoch_num=0, num_classes=3, annealing_step=10, device=device)
loss_late = edl_mse_loss(strong_wrong, target1, epoch_num=20, num_classes=3, annealing_step=10, device=device)
check("annealing_increases_penalty_over_time", loss_late.item() >= loss_early.item())

# 10. edl_predictions returns sane shapes and ranges
out = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 8.0]])
pred_class, confidence, uncertainty = edl_predictions(out)
check(
    "edl_predictions_shapes_and_ranges",
    pred_class.tolist() == [0, 2]
    and torch.all(confidence > 0) and torch.all(confidence <= 1)
    and torch.all(uncertainty >= 0) and torch.all(uncertainty <= 1),
)

print(f"\n{passed}/{total} passed")
