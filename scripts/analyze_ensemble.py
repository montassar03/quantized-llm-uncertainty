"""Load all members of one ensemble and compute ensemble metrics and diversity (Phase 1).

Example:
    python scripts/analyze_ensemble.py --members results/bsr/int5 --reference results/fp/fp32
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics.calibration import accuracy, ece, nll
from metrics.diversity import (ambiguity_decomposition, disagreement, ensemble_mean,
                               mutual_information)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", required=True, help="folder with seed_XX subfolders")
    parser.add_argument("--reference", default="results/fp/fp32", help="full-precision results folder")
    parser.add_argument("--max-members", type=int, default=None, help="use only the first M members")
    args = parser.parse_args()

    files = sorted(Path(args.members).glob("seed_*/probs.npz"))[:args.max_members]
    if not files:
        raise SystemExit(f"No members found in {args.members}")
    data = [np.load(f) for f in files]
    labels = data[0]["labels"]
    for d, f in zip(data, files):
        assert np.array_equal(d["labels"], labels), f"question order differs in {f}"
    P = np.stack([d["probs"] for d in data])          # [M, N, 4]
    p_bar = ensemble_mean(P)

    ref = np.load(Path(args.reference) / "probs.npz")
    assert np.array_equal(ref["labels"], labels), "reference has a different question order"
    p_ref = ref["probs"]

    member_acc = [accuracy(p, labels) for p in P]
    member_nll = [nll(p, labels) for p in P]
    mi = mutual_information(P)
    correct = p_bar.argmax(1) == labels

    report = {
        "n_members": len(files),
        "n_questions": int(len(labels)),
        "reference": {"error": 1 - accuracy(p_ref, labels), "nll": nll(p_ref, labels),
                      "ece": ece(p_ref, labels)},
        "members": {"error_mean": 1 - float(np.mean(member_acc)),
                    "error_std": float(np.std(member_acc)),
                    "nll_mean": float(np.mean(member_nll)),
                    "nll_std": float(np.std(member_nll))},
        "ensemble": {"error": 1 - accuracy(p_bar, labels), "nll": nll(p_bar, labels),
                     "ece": ece(p_bar, labels)},
        "diversity": {**ambiguity_decomposition(P, labels),
                      "pairwise_disagreement": disagreement(P),
                      "flip_rate_vs_reference": float(
                          (P.argmax(2) != p_ref.argmax(1)).mean()),
                      "mean_mutual_information": float(mi.mean()),
                      "mutual_information_wrong": float(mi[~correct].mean()),
                      "mutual_information_right": float(mi[correct].mean())},
    }
    out = Path(args.members) / "ensemble_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()