#!/usr/bin/env python3
"""Resumable Jacobian fit (upper layers) for Qwen3.5-9B SFT, Heretic or base.

Source layers split..30 are fitted against target layer 31. All variants are kept
strictly apart: same corpus, same prompt count, same code path — the only
difference is which checkpoint gets loaded.

    python scripts/qwen35_fit.py --variant sft --n-prompts 1 --tag smoke
    python scripts/qwen35_fit.py --variant sft --n-prompts 20
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_split import DEFAULT_SPLIT, MODEL_IDS, load_split_model, require_free_vram  # noqa: E402

import jlens  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--split", type=int, default=DEFAULT_SPLIT)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus/prompts.json")
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--tag", default="upper", help="name part for checkpoint/lens (e.g. smoke)")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    prompts = json.loads(args.corpus.read_text(encoding="utf-8"))["prompts"][:args.n_prompts]
    require_free_vram()
    t0 = time.time()
    model = load_split_model(args.variant, args.split)
    print(f"Loaded in {time.time() - t0:.0f}s · VRAM {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    source_layers = list(range(args.split, model.n_layers - 1))
    out_dir = ROOT / f"out/qwen35_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"fit_ckpt_{args.tag}.pt"
    print(f"Qwen3.5-9B {args.variant}: {model.n_layers} layers, d_model {model.d_model}, "
          f"source layers {source_layers[0]}..{source_layers[-1]}, {len(prompts)} prompts", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    lens = jlens.fit(
        model, prompts, source_layers=source_layers, dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len, checkpoint_path=str(checkpoint),
        checkpoint_every=1, resume=not args.fresh,
    )
    took = time.time() - t1
    n_done = lens.n_prompts
    lens_out = out_dir / f"jacobian_lens_{args.tag}_{n_done}prompt.pt"
    lens.save(lens_out)
    shutil.copy2(checkpoint, out_dir / f"fit_ckpt_{args.tag}_{n_done}prompt.pt")
    print(f"Fit: {took:.0f}s for {len(prompts)} prompts · peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"Lens saved: {lens_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
