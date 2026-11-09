"""Shared loading code: Qwen3.5-9B (SFT, Heretic or base) with a fixed GPU/CPU layer split.

Same approach as ``gemma_split.py``: layers >= split live permanently in VRAM,
everything below runs on the CPU. For the jlens fit, start_graph_at must be
>= split; then the whole backward pass runs on-device.

Architecture (read from config.json): Qwen3_5ForConditionalGeneration, 32 layers,
d_model 4096, hybrid — three ``linear_attention`` blocks (Gated DeltaNet), then one
``full_attention`` block, repeated. Without ``fla``/``causal_conv1d`` installed the
linear layers fall back to the plain torch implementation (slow, but exact).
The vision tower is loaded, stays on the CPU and is never used.
"""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_ROOT / "hf"))
# Needed on an RX 6700 XT (gfx1031); harmless elsewhere.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

import jlens  # noqa: E402

# Set JLENS_OFFLINE=1 to forbid downloads and use only what is already in HF_HOME.
OFFLINE = os.environ.get("JLENS_OFFLINE", "0") == "1"

MODEL_IDS = {
    "sft": "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B",
    "heretic": "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
    # Third arm. The SFT model card names unsloth/Qwen3.5-9B as base_model, a mirror of this.
    "basis": "Qwen/Qwen3.5-9B",
}
# Heretic edited layers 9–24 (model card). The lens covers L20–30, the upper 12 layers on the GPU.
DEFAULT_SPLIT = 20


def _move_tensors(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def load_split_model(variant: str, split: int = DEFAULT_SPLIT, *, force_bos: bool = True):
    """Load ``variant`` ("sft" | "heretic" | "basis"), move layers >= split to the GPU, wrap for jlens."""
    model_id = MODEL_IDS[variant]
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=OFFLINE)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=None, local_files_only=OFFLINE
    )
    hf_model.eval()

    decoder = hf_model.model.language_model
    layers = decoder.layers
    if not 0 <= split <= len(layers):
        raise ValueError(f"split={split} outside 0..{len(layers)}")

    def to_cuda_hook(module, args, kwargs):
        return _move_tensors(args, "cuda"), _move_tensors(kwargs, "cuda")

    def to_cpu_hook(module, args, kwargs):
        return _move_tensors(args, "cpu"), _move_tensors(kwargs, "cpu")

    for i in range(split, len(layers)):
        layers[i].to("cuda")
        layers[i].register_forward_pre_hook(to_cuda_hook, with_kwargs=True)
    decoder.norm.register_forward_pre_hook(to_cpu_hook, with_kwargs=True)

    return jlens.from_hf(hf_model, tokenizer, force_bos=force_bos)


def require_free_vram(min_gb: float = 9.0) -> None:
    """Fail early instead of dying mid-fit when something else holds the card."""
    free, total = torch.cuda.mem_get_info()
    if free / 1e9 < min_gb:
        raise SystemExit(
            f"Only {free / 1e9:.1f} GB VRAM free (needed: ~{min_gb} GB). "
            "Stop whatever else is using the GPU before fitting."
        )
