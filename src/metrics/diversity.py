
import numpy as np

EPS = 1e-12


def ensemble_mean(P: np.ndarray) -> np.ndarray:
    """Average of member probabilities (Bayesian model average) -> [N, K]."""
    return P.mean(0)


def ensemble_geometric(P: np.ndarray) -> np.ndarray:
    """Normalized geometric mean (softmax of averaged log-probs) -> [N, K].
    This is the combiner for which the cross-entropy ambiguity decomposition is exact
    (Wood et al., 2023); used for the (a)/(b)/(c) diversity analysis in Nam et al. (2024)."""
    log_mean = np.log(P + EPS).mean(0)
    log_mean -= log_mean.max(1, keepdims=True)
    q = np.exp(log_mean)
    return q / q.sum(1, keepdims=True)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + EPS).mean())


def ambiguity_decomposition(P: np.ndarray, labels: np.ndarray) -> dict:
    """(a) average member loss, (c) ensemble loss, (b) ambiguity = (a) - (c)."""
    avg_member_loss = float(np.mean([_nll(p, labels) for p in P]))
    ens_loss = _nll(ensemble_geometric(P), labels)
    return {"avg_member_loss": avg_member_loss,
            "ambiguity": avg_member_loss - ens_loss,
            "ensemble_loss": ens_loss}


def disagreement(P: np.ndarray) -> float:
    """Average pairwise fraction of questions where two members predict different answers."""
    preds = P.argmax(2)                      # [M, N]
    M = len(preds)
    pairs = [(preds[i] != preds[j]).mean() for i in range(M) for j in range(i + 1, M)]
    return float(np.mean(pairs))


def mutual_information(P: np.ndarray) -> np.ndarray:
    """Per-question mutual information (= epistemic part of the entropy decomposition) -> [N]."""
    p_bar = P.mean(0)
    total = -(p_bar * np.log(p_bar + EPS)).sum(1)
    aleatoric = -(P * np.log(P + EPS)).sum(2).mean(0)
    return total - aleatoric