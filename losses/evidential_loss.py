"""
Evidential Deep Learning loss (Sensoy et al., 2018).

Core idea:
    The network outputs "evidence" for each class instead of raw logits.
    evidence = relu(output)              # evidence >= 0
    alpha    = evidence + 1              # Dirichlet concentration params
    S        = sum(alpha)                # total evidence ("Dirichlet strength")
    prob     = alpha / S                 # expected class probabilities
    uncertainty = K / S                  # K = num_classes; less evidence -> more uncertainty

The loss has three parts:
    A - how far predicted probabilities are from the one-hot target
    B - variance term (penalizes overconfident wrong evidence)
    C - KL divergence to the uniform Dirichlet, annealed in over training
        (this term is what discourages evidence for the WRONG classes,
         and is turned on gradually so the model doesn't collapse early)
"""

import torch
import torch.nn.functional as F


def get_target_onehot(
    target: torch.Tensor,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Backward-compatible target adapter with strict soft-target validation.

    Supported target formats:
        [B]    integer class labels
        [B,K]  soft/probability targets (values in [0, 1], row sums == 1.0)
    """
    if target.dim() == 1:
        return F.one_hot(
            target,
            num_classes=num_classes
        ).float().to(device)

    elif target.dim() == 2:
        if target.shape[1] != num_classes:
            raise ValueError(
                f"Expected target shape [B, {num_classes}], "
                f"got {tuple(target.shape)}"
            )

        target = target.float().to(device)

        if torch.any(target < 0.0) or torch.any(target > 1.0):
            raise ValueError("Soft targets must lie within [0.0, 1.0].")

        row_sums = target.sum(dim=1)
        ones = torch.ones(target.shape[0], device=device)
        if not torch.allclose(row_sums, ones, atol=1e-5):
            raise ValueError("Each soft-target row must sum to 1.0.")

        return target

    else:
        raise ValueError(
            "Target tensor must be 1D [B] or "
            f"2D [B, K], got shape {tuple(target.shape)}"
        )


def relu_evidence(y: torch.Tensor) -> torch.Tensor:
    """Map raw network outputs to non-negative 'evidence'."""
    return F.relu(y)


def kl_divergence(alpha: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ), i.e. distance from the uniform prior."""
    beta = torch.ones((1, num_classes), dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def edl_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int,
    num_classes: int,
    annealing_step: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Args:
        output: raw model output, shape (batch, num_classes) -- NOT softmax'd
        target: integer class labels (batch,) or soft targets (batch, num_classes)
        epoch_num: current epoch (used to anneal the KL term in slowly)
        num_classes: number of classes K
        annealing_step: epoch at which the KL weight reaches 1.0
        device: torch device
    """
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    m = alpha / S

    target_onehot = get_target_onehot(target, num_classes, device)

    # A: squared error between predicted expected prob and target matrix
    A = torch.sum((target_onehot - m) ** 2, dim=1, keepdim=True)
    # B: expected variance of the Dirichlet -- penalizes evidence spread across wrong classes
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    annealing_coef = min(1.0, float(epoch_num) / float(annealing_step))

    # Remove evidence for the target classes before computing KL
    alpha_tilde = evidence * (1 - target_onehot) + 1
    C = annealing_coef * kl_divergence(alpha_tilde, num_classes, device)

    return torch.mean(A + B + C)


def edl_predictions(output: torch.Tensor):
    """
    Convert raw model output into (predicted_class, confidence, uncertainty).
    confidence = max class probability (alpha_k / S)
    uncertainty = K / S  (in [0, 1], higher = more uncertain)
    """
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    num_classes = output.shape[1]

    confidence, pred_class = torch.max(prob, dim=1)
    uncertainty = (num_classes / S).squeeze(1)
    return pred_class, confidence, uncertainty


def redl_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    epoch_num: int,
    num_classes: int,
    annealing_step: int,
    device: torch.device,
    lam: float = 0.1,
) -> torch.Tensor:
    """
    R-EDL loss based on Chen, Gao, and Xu, ICLR 2024.
    """
    evidence = relu_evidence(output)
    alpha = evidence + lam                        # Eq. 9: generalized alpha
    S = torch.sum(alpha, dim=1, keepdim=True)
    P = alpha / S                                  # projected probability

    target_onehot = get_target_onehot(target, num_classes, device)

    # Direct squared error on projected probability.
    L_redl = torch.sum((target_onehot - P) ** 2, dim=1, keepdim=True)

    annealing_coef = min(1.0, float(epoch_num) / float(max(1, annealing_step)))

    # KL regularization: target class -> 1, non-target classes -> alpha
    alpha_tilde = alpha * (1 - target_onehot) + target_onehot
    C = annealing_coef * kl_divergence(alpha_tilde, num_classes, device)

    return torch.mean(L_redl + C)


def redl_predictions(output: torch.Tensor, lam: float = 0.1):
    """
    R-EDL prediction and uncertainty.
    """
    evidence = relu_evidence(output)
    alpha = evidence + lam
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    num_classes = output.shape[1]

    confidence, pred_class = torch.max(prob, dim=1)
    uncertainty = (lam * num_classes / S).squeeze(1)   # authors' correction
    return pred_class, confidence, uncertainty