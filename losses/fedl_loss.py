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

        self.g_alpha = nn.Linear(in_features, num_classes)
        self.g_p = nn.Linear(in_features, num_classes)
        self.g_tau = nn.Linear(in_features, 1)

    def forward(self, features):
        # Prevent exp() from producing inf/nan.
        log_alpha = torch.clamp(
            self.g_alpha(features),
            min=-20.0,
            max=20.0,
        )

        alpha = torch.exp(log_alpha)

        p = F.softmax(self.g_p(features), dim=1)

        tau = F.softplus(self.g_tau(features))

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

    alpha0 = alpha.sum(dim=1, keepdim=True)

    denom = alpha0 + tau

    mu = (alpha + tau * p) / denom

    var = (
        mu * (1.0 - mu) / (denom + 1.0)
        + (
            tau.pow(2)
            * p
            * (1.0 - p)
            / (denom * (denom + 1.0))
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

    L =
        E[||y - pi||^2]
        +
        ||y - p||^2
    """

    y = F.one_hot(
        target,
        num_classes=num_classes,
    ).float()

    mu, var = compute_moments(
        alpha,
        p,
        tau,
    )

    # Expected squared classification error
    classification_loss = (
        (y - mu).pow(2) + var
    ).sum(dim=1)

    # Allocation/Brier-style regularization
    allocation_loss = (
        y - p
    ).pow(2).sum(dim=1)

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

    confidence, pred_class = mu.max(dim=1)

    # Expected uncertainty
    epistemic_uncertainty = var.sum(dim=1)

    # Total uncertainty
    total_uncertainty = (
        1.0 - mu.pow(2).sum(dim=1)
    )

    # Aleatoric component
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