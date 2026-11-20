"""Create quantized ensemble members (RTN or LPE-BSR) and evaluate each on MMLU.

Each member is built from the original weights with its own seed and saved to
<out-root>/<mode>/int<bits>/seed_XX/{probs.npz, metrics.json}. Existing members are skipped,
so a job that hits its time limit can simply be resubmitted.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.mmlu import evaluate_mmlu, load_mmlu
from metrics.calibration import accuracy, ece, nll
from quantization.lpe_bsr import make_member, save_original_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to the model folder")
    parser.add_argument("--mode", default="bsr", choices=["bsr", "rtn"])
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing)")
    parser.add_argument("--originals-device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # RTN is deterministic: one member is enough
    seeds = [0] if args.mode == "rtn" else list(range(args.seed_start, args.seed_start + args.num_seeds))
    out_dir = Path(args.out_root) / args.mode / f"int{args.bits}"
    todo = [s for s in seeds if not (out_dir / f"seed_{s:02d}" / "probs.npz").exists()]
    if not todo:
        print("All requested members already exist:", seeds)
        return
    print("Members to run:", todo)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    model.eval()
    originals = save_original_weights(model, device=args.originals_device)

    dataset = load_mmlu(limit=args.limit)
    labels = np.array(dataset["answer"])

    for seed in todo:
        t0 = time.time()
        make_member(model, originals, bits=args.bits, mode=args.mode, seed=seed)
        t_quant = time.time() - t0

        probs, mass = evaluate_mmlu(model, tokenizer, dataset, batch_size=args.batch_size)

        metrics = {
            "mode": args.mode, "bits": args.bits, "seed": seed,
            "dtype": args.dtype, "batch_size": args.batch_size,
            "n_questions": int(len(labels)),
            "accuracy": accuracy(probs, labels),
            "error": 1 - accuracy(probs, labels),
            "nll": nll(probs, labels),
            "ece": ece(probs, labels),
            "mean_mass_on_ABCD": float(mass.mean()),
            "min_mass_on_ABCD": float(mass.min()),
            "seconds_quantize": round(t_quant, 1),
            "seconds_total": round(time.time() - t0, 1),
        }
        member_dir = out_dir / f"seed_{seed:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        np.savez(member_dir / "probs.npz", probs=probs, labels=labels, mass=mass,
                 subjects=np.array(dataset["subject"]))
        (member_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()