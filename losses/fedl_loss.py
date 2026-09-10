import torch
import torch.nn as nn
import torch.nn.functional as F


class FEDLHead(nn.Module):
    """
    Flexible Evidential Deep Learning head.

    Outputs:
        alpha : [B, K]
        p     : [B, K]
        tau   : [B, 1]
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.g_alpha = nn.Linear(
            in_features,
            num_classes,
        )

        self.g_p = nn.Linear(
            in_features,
            num_classes,
        )

        self.g_tau = nn.Linear(
            in_features,
            1,
        )

    def forward(self, features):
        """
        Forward pass through the F-EDL head.
        """

        # Prevent exp() from producing inf/nan.
        log_alpha = torch.clamp(
            self.g_alpha(features),
            min=-20.0,
            max=20.0,
        )

        alpha = torch.exp(log_alpha)

        # Probability allocation vector.
        p = F.softmax(
            self.g_p(features),
            dim=1,
        )

        # Shared uncertainty/concentration parameter.
        tau = F.softplus(
            self.g_tau(features),
        )

        return alpha, p, tau


def compute_moments(alpha, p, tau):
    """
    Compute the mean and variance of the Flexible Dirichlet.

    Args:
        alpha: [B, K]
        p:     [B, K]
        tau:   [B, 1]

    Returns:
        mu:  expected class probabilities [B, K]
        var: class-wise variance [B, K]
    """

    alpha0 = alpha.sum(
        dim=1,
        keepdim=True,
    )

    denom = alpha0 + tau

    # Expected class probability:
    #
    # mu_k = (alpha_k + tau * p_k) / (alpha_0 + tau)
    mu = (
        alpha + tau * p
    ) / denom

    # Class-wise variance.
    var = (
        mu * (1.0 - mu) / (denom + 1.0)
        +
        (
            tau.pow(2)
            * p
            * (1.0 - p)
            / (
                denom
                * (denom + 1.0)
            )
        )
    )

    return mu, var


def fedl_loss(
    alpha,
    p,
    tau,
    target,
    num_classes,
):
    """
    F-EDL training objective.

    Supports both:

        1. Integer class labels:
           target shape = [B]

        2. Soft/continuous targets:
           target shape = [B, K]

    The second form is required for Mixup and CutMix.

    F-EDL objective:

        L =
            E[||y - pi||^2]
            +
            ||y - p||^2
    """

    # ---------------------------------------------------------
    # Target handling
    # ---------------------------------------------------------

    if target.dim() == 1:

        # Standard classification labels:
        # [B] -> [B, K]
        y = F.one_hot(
            target.long(),
            num_classes=num_classes,
        ).float()

    elif target.dim() == 2:

        # Mixup / CutMix soft targets:
        # already [B, K]
        y = target.float()

    else:
        raise ValueError(
            "Invalid target dimensions for F-EDL. "
            f"Expected [B] or [B, K], got {tuple(target.shape)}."
        )

    # ---------------------------------------------------------
    # Target shape verification
    # ---------------------------------------------------------

    expected_shape = (
        alpha.size(0),
        num_classes,
    )

    if y.shape != expected_shape:
        raise ValueError(
            "Target shape mismatch in F-EDL loss. "
            f"Expected {expected_shape}, "
            f"got {tuple(y.shape)}."
        )

    # ---------------------------------------------------------
    # Flexible Dirichlet moments
    # ---------------------------------------------------------

    mu, var = compute_moments(
        alpha,
        p,
        tau,
    )

    # ---------------------------------------------------------
    # Expected squared classification error
    # ---------------------------------------------------------

    classification_loss = (
        (y - mu).pow(2) + var
    ).sum(
        dim=1
    )

    # ---------------------------------------------------------
    # Allocation / Brier-style regularization
    # ---------------------------------------------------------

    allocation_loss = (
        y - p
    ).pow(2).sum(
        dim=1
    )

    # ---------------------------------------------------------
    # Total F-EDL loss
    # ---------------------------------------------------------

    loss = (
        classification_loss
        + allocation_loss
    ).mean()

    return loss


def fedl_predictions(alpha, p, tau):
    """
    Convert F-EDL outputs into prediction and uncertainty quantities.

    Returns:
        pred_class
        confidence
        total_uncertainty
        epistemic_uncertainty
        aleatoric_uncertainty
    """

    mu, var = compute_moments(
        alpha,
        p,
        tau,
    )

    # Predicted class and confidence.
    confidence, pred_class = mu.max(
        dim=1
    )

    # Epistemic uncertainty.
    epistemic_uncertainty = var.sum(
        dim=1
    )

    # Total uncertainty.
    total_uncertainty = (
        1.0
        - mu.pow(2).sum(dim=1)
    )

    # Aleatoric uncertainty.
    #
    # Total = Epistemic + Aleatoric
    #
    # Numerical protection is applied because
    # floating-point calculations can occasionally
    # produce tiny negative values.
    aleatoric_uncertainty = torch.clamp(
        total_uncertainty
        - epistemic_uncertainty,
        min=0.0,
    )

    return (
        pred_class,
        confidence,
        total_uncertainty,
        epistemic_uncertainty,
        aleatoric_uncertainty,
    )