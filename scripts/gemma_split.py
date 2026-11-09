"""Shared loading code: Gemma 4 12B in bf16 with a fixed GPU/CPU layer split.

Layers >= split live permanently in VRAM. No accelerate offload — that is
forward-only: its hooks evict the weights once the forward pass is done, so the
autograd graph is already torn by the time the backward starts. Pre-hooks hand the
GPU layers their inputs/kwargs on the card; the final norm pulls back to CPU.

For the jlens fit, start_graph_at must be >= split. Then the autograd root itself
sits on the GPU and the entire backward runs on-device — the CPU->GPU transition
happens before the root and is invisible to autograd. That single detail is what
takes the backward pass from 17 minutes to 1.2 seconds.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # repo root


import os

os.environ.setdefault("HF_HOME", str(_ROOT / "hf"))
# RX 6700 XT is gfx1031, which ROCm does not ship kernels for; 10.3.0 (gfx1030) works.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

MODEL_ID = "google/gemma-4-12B-it"
DEFAULT_SPLIT = 33

# Set JLENS_OFFLINE=1 to forbid downloads and use only what is already in HF_HOME.
OFFLINE = os.environ.get("JLENS_OFFLINE", "0") == "1"


def _move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def load_split_model(split: int = DEFAULT_SPLIT, *, force_bos: bool = True):
    """Load the model, move layers >= split onto the GPU, wrap it for jlens.

    For raw text prompts ``jlens`` prepends a BOS token by default. That must not
    happen for an already-rendered chat template, because Gemma's template starts
    with ``<bos>`` itself — you would end up with two. Callers that feed rendered
    templates pass ``force_bos=False`` and check the token sequence before the
    forward pass.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=OFFLINE)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=None, local_files_only=OFFLINE
    )

    decoder = hf_model.model.language_model
    layers = decoder.layers

    def to_cuda_hook(module, args, kwargs):
        return _move_tensors(args, "cuda"), _move_tensors(kwargs, "cuda")

    def to_cpu_hook(module, args, kwargs):
        return _move_tensors(args, "cpu"), _move_tensors(kwargs, "cpu")

    for i in range(split, len(layers)):
        layers[i].to("cuda")
        layers[i].register_forward_pre_hook(to_cuda_hook, with_kwargs=True)
    decoder.norm.register_forward_pre_hook(to_cpu_hook, with_kwargs=True)

    return jlens.from_hf(hf_model, tokenizer, force_bos=force_bos)


def require_free_vram(min_gb: float = 11.0) -> None:
    """Check up front rather than dying 14 minutes into a fit: the fit needs ~11.4 GB,
    and an inference server sitting on the card will happily take that away."""
    free, total = torch.cuda.mem_get_info()
    if free / 1e9 < min_gb:
        raise SystemExit(
            f"Only {free / 1e9:.1f} GB VRAM free (needed: ~{min_gb} GB). "
            "Something else is probably holding the card — e.g. ollama: "
            "systemctl stop ollama"
        )
