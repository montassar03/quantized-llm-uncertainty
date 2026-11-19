"""Evaluate the full-precision model on MMLU and save per-question probabilities."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.mmlu import evaluate_mmlu, load_mmlu
from metrics.calibration import accuracy, ece, nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to the model folder")
    parser.add_argument("--out", default="results/fp/fp32", help="output folder")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    model.eval()

    dataset = load_mmlu(limit=args.limit)
    labels = np.array(dataset["answer"])

    start = time.time()
    probs, mass = evaluate_mmlu(model, tokenizer, dataset, batch_size=args.batch_size)

    metrics = {
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "n_questions": int(len(labels)),
        "accuracy": accuracy(probs, labels),
        "error": 1 - accuracy(probs, labels),
        "nll": nll(probs, labels),
        "ece": ece(probs, labels),
        "mean_mass_on_ABCD": float(mass.mean()),
        "min_mass_on_ABCD": float(mass.min()),
        "seconds": round(time.time() - start, 1),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "probs.npz", probs=probs, labels=labels, mass=mass,
             subjects=np.array(dataset["subject"]))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()