"""Every number in the README, in one run. No GPU, no model weights.

Inputs: the raw results in out/ and the manual review in corpus/facts/manual_review.json.
The script checks that every override actually changes a verdict and that every
confabulation has a recorded core (and vice versa), so the hand verdicts cannot
silently drift away from the data.

    python scripts/metrics.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_content_tokens  # noqa: E402
import analyze_h4  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FACTS = {m: ROOT / p for m, p in analyze_content_tokens.FILES.items()}
SIGNALS = {m: ROOT / f"out/qwen35_{m}/refusal_signals_jacobian_lens_upper_20prompt.json"
           for m in ("sft", "heretic", "basis")}
REVIEW = json.load(open(ROOT / "corpus/facts/manual_review.json"))
GOOD = {"fiktiv": "nichtwissen", "fakt": "richtig", "mathe": "richtig"}
CATEGORY = {"fiktiv": "fictional (abstained)", "fakt": "fact (correct)", "mathe": "math (correct)"}


def load():
    items = {m: {i["id"]: i for i in json.load(open(p))["items"]} for m, p in FACTS.items()}
    hand = {m: {k: i["urteil"] for k, i in its.items()} for m, its in items.items()}
    for o in REVIEW["overrides"]:
        assert o["verdict"] != items[o["model"]][o["id"]]["urteil"], f"override without effect: {o}"
        hand[o["model"]][o["id"]] = o["verdict"]
    for m, verdicts in hand.items():
        konf = {k for k, v in verdicts.items() if v == "konfabuliert"}
        cores = set(REVIEW["cores"].get(m, {}))
        assert konf == cores, f"{m}: cores {sorted(cores)} != confabulated {sorted(konf)}"
    return items, hand


def scores(items, hand):
    print("== Scores (automatic -> manual review)")
    print(f"   {'':22}" + "".join(f"{m:>11}" for m in FACTS))
    for cat in ("fiktiv", "fakt", "mathe"):
        cells = []
        for m in FACTS:
            ids = [k for k, i in items[m].items() if i["kat"] == cat]
            auto = sum(items[m][k]["urteil"] == GOOD[cat] for k in ids)
            manual = sum(hand[m][k] == GOOD[cat] for k in ids)
            cells.append(f"{auto}->{manual}/{len(ids)}")
        print(f"   {CATEGORY[cat]:22}" + "".join(f"{c:>11}" for c in cells))
    print(f"   overrides: {len(REVIEW['overrides'])}")


def identical(items):
    s, h = items["sft"], items["heretic"]
    same = [k for k in s if s[k]["antwort"].strip() == h[k]["antwort"].strip()]
    flipped = [k for k in s if s[k]["urteil"] != h[k]["urteil"]]
    print(f"\n== SFT vs Heretic: {len(same)}/{len(s)} answers word-for-word identical; "
          f"automatic verdict differs on {flipped or 'none'}")


def shared_inventions():
    cores = REVIEW["cores"]
    print("\n== Same invention (cores from manual_review.json, items confabulated by both)")
    for a, b in combinations(cores, 2):
        both = sorted(set(cores[a]) & set(cores[b]))
        if both:
            same = [k for k in both if cores[a][k] == cores[b][k]]
            print(f"   {a:7} vs {b:7} {len(same)}/{len(both)} identical  {same}")


def refusals():
    print("\n== Refusals (keyword rule, 32 tokens)")
    for m, p in SIGNALS.items():
        d = json.load(open(p))["daten"]
        parts = [f"{g} {sum(r['abgelehnt'] for r in d[g]['zeilen'])}/{len(d[g]['zeilen'])}" for g in d]
        print(f"   {m:8}", "  ".join(parts))


def main():
    items, hand = load()
    scores(items, hand)
    identical(items)
    shared_inventions()
    refusals()
    print("\n== H2 / H5 / Gemma: content tokens")
    analyze_content_tokens.main()
    print("\n== H4")
    analyze_h4.main()


if __name__ == "__main__":
    main()
