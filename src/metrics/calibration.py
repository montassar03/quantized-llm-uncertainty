"""First-order metrics of a predictive distribution: accuracy, NLL, ECE."""
import numpy as np


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(1) == labels).mean())


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on the top-class confidence."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)