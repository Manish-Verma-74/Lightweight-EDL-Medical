"""
Expected Calibration Error (ECE) and reliability diagrams.
Based on Guo et al., 2017, "On Calibration of Modern Neural Networks".
"""
import numpy as np


def compute_ece(confidences, predictions, labels, n_bins: int = 15):
    """
    Args:
        confidences: array of predicted-class confidences, shape (N,), in [0, 1]
        predictions: array of predicted class indices, shape (N,)
        labels: array of true class indices, shape (N,)
        n_bins: number of equal-width confidence bins

    Returns:
        ece: float, the expected calibration error
        bin_data: list of (lower, upper, accuracy_in_bin, avg_confidence_in_bin, fraction_of_samples)
                  for each non-empty bin -- used to draw the reliability diagram
    """
    confidences = np.asarray(confidences, dtype=float)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []
    n = len(confidences)
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = np.sum(in_bin) / n if n > 0 else 0.0
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_data.append((lower, upper, acc_in_bin, conf_in_bin, prop_in_bin))

    return float(ece), bin_data


def plot_reliability_diagram(bin_data, n_bins: int, save_path: str = None, title: str = "Reliability Diagram"):
    """
    Draws a reliability diagram (accuracy vs confidence bars + perfect-calibration line).

    bin_data only contains NON-EMPTY bins (see compute_ece), but we still draw all
    `n_bins` slots at a fixed width -- empty bins are drawn at zero height. This keeps
    bar width constant and the x-axis correctly interpretable as equal-width confidence
    intervals, matching how reliability diagrams are drawn in published papers.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    width = 1.0 / n_bins

    # Map each non-empty bin's accuracy onto its bin index by matching lower edge
    accs_by_bin = np.zeros(n_bins)
    tol = width / 2  # tolerance for float comparison when matching edges
    for lower, upper, acc, conf, frac in bin_data:
        idx = int(np.argmin(np.abs(bin_lowers - lower)))
        if abs(bin_lowers[idx] - lower) < tol:
            accs_by_bin[idx] = acc

    ax.bar(bin_lowers, accs_by_bin, width=width, align="edge", edgecolor="black",
           color="#4C72B0", label="Model accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig