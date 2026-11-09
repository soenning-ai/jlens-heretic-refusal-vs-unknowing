"""H4: lens entropy of the first k answer tokens, Heretic − SFT (and base − SFT).

Reads only numbers from out/qwen35_<v>/refusal_signals_jacobian_lens_upper_20prompt.json
(no token strings, no prompts exist in these files).

The pre-registration did not fix k, so all three values are reported — that is a
degree of freedom and is named as one.

    python scripts/analyze_h4.py          # mean per prompt over tokens[:k] and layers, then over prompts
    python scripts/analyze_h4.py --pool   # all (prompt, token, layer) values pooled
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = "out/qwen35_{}/refusal_signals_jacobian_lens_upper_20prompt.json"
RANGES = {"L20–30": [str(x) for x in range(20, 31)], "L27–30": [str(x) for x in range(27, 31)]}


def value(model, group, k, layers, pool=False):
    rows = json.load(open(ROOT / FILE.format(model)))["daten"][group]["zeilen"]
    if pool:
        return st.mean(t["H"][x] for r in rows for t in r["tokens"][:k] for x in layers)
    return st.mean(st.mean(t["H"][x] for t in r["tokens"][:k] for x in layers)
                   for r in rows if r["tokens"])


def main():
    pool = "--pool" in sys.argv
    print("mode:", "pooled" if pool else "per prompt")
    print(f"{'k':>2} {'layers':7} {'Δ harmful':>10} {'Δ harmless':>11} {'base−SFT harmful':>17}")
    for k in (1, 3, 5):
        for name, layers in RANGES.items():
            s = {g: value("sft", g, k, layers, pool) for g in ("harmful", "harmless")}
            h = {g: value("heretic", g, k, layers, pool) for g in ("harmful", "harmless")}
            b = value("basis", "harmful", k, layers, pool)
            print(f"{k:>2} {name:7} {h['harmful'] - s['harmful']:>+10.3f} "
                  f"{h['harmless'] - s['harmless']:>+11.3f} {b - s['harmful']:>+17.3f}")


if __name__ == "__main__":
    main()
