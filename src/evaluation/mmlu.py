"""MMLU evaluation: original zero-shot prompt format, probabilities over A/B/C/D."""
import torch
from datasets import load_dataset
from tqdm import tqdm

LETTERS = ["A", "B", "C", "D"]


def format_subject(subject: str) -> str:
    # "high_school_physics" -> "high school physics"
    return " ".join(subject.split("_"))


def build_prompt(example) -> str:
    """Zero-shot prompt in the original MMLU format (Hendrycks et al., 2021)."""
    prompt = ("The following are multiple choice questions (with answers) "
              f"about {format_subject(example['subject'])}.\n\n")
    prompt += example["question"]
    for letter, choice in zip(LETTERS, example["choices"]):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    return prompt


def load_mmlu(split: str = "test", limit: int | None = None):
    ds = load_dataset("cais/mmlu", "all", split=split)
    if limit is not None:
        ds = ds.select(range(limit))
    return ds


def option_token_ids(tokenizer) -> list[int]:
    """Token ids of " A", " B", " C", " D" (with leading space: text continues after 'Answer:')."""
    ids = []
    for letter in LETTERS:
        tok = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(tok) != 1:
            raise ValueError(f"' {letter}' is not a single token: {tok}")
        ids.append(tok[0])
    return ids


@torch.inference_mode()
def evaluate_mmlu(model, tokenizer, dataset, batch_size: int = 8, device: str = "cuda"):
    """Returns probs [N, 4] (softmax over A-D) and mass [N] (prob. of A-D in the full vocabulary)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # last position = last real token for every prompt
    option_ids = torch.tensor(option_token_ids(tokenizer), device=device)

    prompts = [build_prompt(ex) for ex in dataset]
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))   # similar lengths per batch -> less padding
    probs = torch.empty(n, len(LETTERS))
    mass = torch.empty(n)

    for start in tqdm(range(0, n, batch_size), desc="MMLU"):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        # with left padding, positions must start at 0 at the first real token
        position_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        position_ids=position_ids,
                        logits_to_keep=1).logits[:, -1, :].float()
        mass[idx] = torch.softmax(logits, dim=-1)[:, option_ids].sum(-1).cpu()
        probs[idx] = torch.softmax(logits[:, option_ids], dim=-1).cpu()

    return probs.numpy(), mass.numpy()