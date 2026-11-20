"""Per-channel INT-B quantization with round-to-nearest (RTN) or Bernoulli stochastic
rounding (BSR), following Nam et al. (2024), Eq. 1-2. Simulated ("fake") quantization:
weights get exactly the quantized values but are stored back in the model's dtype.

Every member is built from the ORIGINAL weights (never from a previous member):
    originals = save_original_weights(model)
    for seed in seeds:
        make_member(model, originals, bits=5, mode="bsr", seed=seed)
        ... evaluate ...
    restore_original_weights(model, originals)
"""
import torch
import torch.nn as nn

# Not quantized (as in the official LPE-BSR code): input embeddings and output head.
SKIP_MODULES = ("embed_tokens", "lm_head")


def _quantized_linears(model: nn.Module):
    """All linear layers that get quantized, in a fixed order."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in name for s in SKIP_MODULES):
            yield name, module


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int, mode: str = "bsr",
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """Quantize a weight matrix w [out, in] onto a symmetric per-output-channel INT-B grid."""
    w32 = w.float()                                   # quantize in FP32 (no extra rounding)
    qmax = 2 ** (bits - 1) - 1                        # e.g. INT-5 -> 15
    scale = w32.abs().amax(dim=1, keepdim=True) / qmax    # grid step Delta_i per output row
    scale = scale.clamp(min=1e-12)                    # guard against all-zero rows
    x = w32 / scale                                   # weights in grid units, in [-qmax, qmax]

    if mode == "rtn":
        q = torch.round(x)
    elif mode == "bsr":
        low = torch.floor(x)
        p_up = x - low                                # lambda: probability of rounding up
        u = torch.rand(x.shape, generator=generator)  # uniform [0,1), drawn on CPU (reproducible)
        q = low + (u.to(x.device) < p_up).float()
    else:
        raise ValueError(f"unknown mode: {mode}")

    q = q.clamp(-qmax, qmax)
    return (q * scale).to(w.dtype)


@torch.no_grad()
def save_original_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the full-precision weights of all quantized layers to CPU (call once, before any member)."""
    return {name: module.weight.detach().to("cpu", copy=True)
            for name, module in _quantized_linears(model)}


@torch.no_grad()
def make_member(model: nn.Module, originals: dict[str, torch.Tensor],
                bits: int, mode: str = "bsr", seed: int = 0) -> nn.Module:
    """Turn `model` into member `seed`, always starting from the ORIGINAL weights."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name, module in _quantized_linears(model):
        w = originals[name].to(module.weight.device)
        module.weight.copy_(quantize_weight(w, bits, mode, generator))
    return model


@torch.no_grad()
def save_original_weights(model: nn.Module, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().to(device, copy=True)
            for name, module in _quantized_linears(model)}