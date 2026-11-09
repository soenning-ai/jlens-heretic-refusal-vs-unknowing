#!/usr/bin/env python3
"""Fact battery: generate an answer, then read the lens at every answer token.

Per item:
  1. Greedy answer (chat template, thinking disabled, max. 48 new tokens)
  2. One forward pass over prompt + answer, lens applied at every position that
     predicts an answer token
  3. Per answer token and layer: entropy (nats), rank of the emitted token,
     top-3 (token, p). Lock-in layer ("kipp") = lowest fitted layer from which the
     emitted token stays rank 1 all the way up (None = only in the output)
  4. Automatic verdict: fictional -> correct only if the model abstains (regex
     ABSTAIN); fact/math -> accepted substring. Every answer was then reviewed by
     hand, see corpus/facts/manual_review.json

The output keeps the original (German) field names, see out/README.md.

    python scripts/qwen35_facts_battery.py --variant sft --lens out/qwen35_sft/jacobian_lens_upper_20prompt.pt
    python scripts/qwen35_facts_battery.py --variant gemma4 --lens <gemma 100-prompt lens>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_split import DEFAULT_SPLIT, MODEL_IDS, load_split_model, require_free_vram  # noqa: E402

import torch  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "corpus/facts/battery.json"
ABSTAIN = re.compile(
    r"(don't know|do not know|not aware|no (?:reliable |known |widely known )?(?:information|record|evidence)|"
    r"does not exist|doesn't exist|not exist|fictional|fictitious|not familiar|unable to find|"
    r"couldn't find|could not find|no known|not a (?:real|known|recognized)|i'm not sure|i am not sure|"
    r"cannot find|can't find|not recognize|unfamiliar)",
    re.IGNORECASE,
)


def judge(item: dict, answer: str) -> dict:
    if item["kat"] == "fiktiv":
        abstained = bool(ABSTAIN.search(answer))
        return {"richtig": abstained, "urteil": "nichtwissen" if abstained else "konfabuliert"}
    hit = any(a.lower() in answer.lower() for a in item["accept"])
    return {"richtig": hit, "urteil": "richtig" if hit else "falsch"}


def entropy(logits: torch.Tensor) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(-(lp.exp() * lp).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=(*MODEL_IDS, "gemma4"), required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--split", type=int, default=None, help="default: Qwen 20, Gemma 4 33")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--only", nargs="*", help="only these item ids")
    args = ap.parse_args()

    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    items = [i for i in battery["items"] if not args.only or i["id"] in args.only]
    lens = JacobianLens.load(str(args.lens))
    require_free_vram()
    if args.variant == "gemma4":   # cross-check: Gemma 4 12B with the existing 100-prompt lens
        from gemma_split import DEFAULT_SPLIT as G_SPLIT, MODEL_ID as G_ID, load_split_model as g_load
        model = g_load(args.split or G_SPLIT, force_bos=False)
        model_id, out_dir = G_ID, ROOT / "out/gemma4_12b"
    else:
        model = load_split_model(args.variant, args.split or DEFAULT_SPLIT, force_bos=False)
        model_id, out_dir = MODEL_IDS[args.variant], ROOT / f"out/qwen35_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, hf = model.tokenizer, model._hf_model
    layers = sorted(lens.source_layers)

    results = []
    for item in items:
        t0 = time.time()
        msgs = [{"role": "system", "content": battery["system"]}, {"role": "user", "content": item["q"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            gen = hf.generate(input_ids=p_ids, max_new_tokens=args.max_new, do_sample=False)
        a_ids = gen[0, p_ids.shape[1]:].tolist()
        while a_ids and a_ids[-1] in (tok.eos_token_id, tok.pad_token_id):
            a_ids.pop()
        answer = tok.decode(a_ids, skip_special_tokens=True).strip()

        full = prompt + tok.decode(a_ids, skip_special_tokens=False)
        ids = tok(full, add_special_tokens=False).input_ids
        P = p_ids.shape[1]
        prefix_ok = ids[:P] == p_ids[0].tolist() and ids[P:] == a_ids
        positions = list(range(P - 1, P - 1 + len(a_ids)))
        jl, out_logits, _ = lens.apply(model, full, positions=positions, max_seq_len=len(ids) + 8)

        tokens = []
        for k, t_id in enumerate(a_ids):
            probes = {}
            for layer in layers + ["out"]:
                lg = out_logits[k] if layer == "out" else jl[layer][k]
                rank = int((lg > lg[t_id]).sum())
                top = torch.softmax(lg.float(), -1).topk(3)
                probes[str(layer)] = {
                    "H": round(entropy(lg), 4), "rang": rank,
                    "top3": [[tok.decode([int(i)]), round(float(p), 4)] for p, i in zip(top.values, top.indices)],
                }
            lock_in = None
            for layer in reversed(layers):
                if probes[str(layer)]["rang"] == 0:
                    lock_in = layer
                else:
                    break
            tokens.append({"token": tok.decode([t_id]), "kipp": lock_in, "layer": probes})

        e = {"id": item["id"], "kat": item["kat"], "q": item["q"], "antwort": answer,
             **judge(item, answer), "praefix_ok": prefix_ok, "tokens": tokens,
             "sek": round(time.time() - t0, 1)}
        results.append(e)
        print(f"{item['id']} {e['urteil']:<12} {e['sek']:>6}s  {answer[:90]!r}", flush=True)

    out = out_dir / f"facts_{args.lens.stem}.json"
    out.write_text(json.dumps({"variant": args.variant, "model": model_id,
                               "lens": str(args.lens), "layers": layers, "items": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
