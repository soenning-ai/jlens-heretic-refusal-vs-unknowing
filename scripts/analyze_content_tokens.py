"""H2 / H5 and the Gemma cross-check: lens entropy and lock-in at hand-marked content tokens.

The content token is the first token of the core claim of an answer:
  - confabulated fictional items: the first token of the invented claim, marked by
    hand with a regex per model (KONF below)
  - correct fact items: the first token of the accepted answer; for years the first
    informative digit (1648 -> "6")
Items where a denial is followed by an invention (SFT/Heretic f05, base f04/f05) are
left out of the confabulated group: there is no single invented claim to point at.
The base model's t14 is wrong by manual review and is left out of the fact group.

Lock-in "None" (the token is never rank 1 inside the fitted layers) counts as
last fitted layer + 1 (Qwen: 31, Gemma: 47).

    python scripts/analyze_content_tokens.py
"""
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATTERY = {i["id"]: i for i in json.load(open(ROOT / "corpus/facts/battery.json"))["items"]}

KONF_SFT = {"f02": r"(John)", "f04": r"(Mir)iam", "f07": r"(ty)ros", "f08": r"1(6)59",
            "f09": r"(H)ugo", "f11": r"founded by \**\s*(\w)"}
KONF = {
    "sft": KONF_SFT,
    "heretic": KONF_SFT,
    "basis": {"f07": r"inhibitor of the (\w)", "f08": r"1(8)14", "f09": r"\*\*(H)ans",
              "f11": r"linguist (T)"},
    "gemma": {"f09": r"(H)ugo"},
}
FACT_EXCLUDED = {"basis": {"t14"}}
SHARED = ("f07", "f08", "f09", "f11")  # confabulated and marked in all three Qwen models
FILES = {
    "sft": "out/qwen35_sft/facts_jacobian_lens_upper_20prompt.json",
    "heretic": "out/qwen35_heretic/facts_jacobian_lens_upper_20prompt.json",
    "basis": "out/qwen35_basis/facts_jacobian_lens_upper_20prompt.json",
    "gemma": "out/gemma4_12b/facts_jacobian_lens_upper.json",
}


def pos(item, core_re=None):
    """Index of the answer token that contains the core claim."""
    txt = ""
    starts = []
    for t in item["tokens"]:
        starts.append(len(txt))
        txt += t["token"]
    if core_re:
        m = re.search(core_re, txt)
        if not m:
            return None, txt
        c = m.start(1)
    else:
        low = txt.lower()
        c, a = min((low.find(a.lower()), a) for a in BATTERY[item["id"]]["accept"] if a.lower() in low)
        if re.fullmatch(r"1\d{3}", a):
            c += 1  # year: first informative digit
    return max(j for j, s in enumerate(starts) if s <= c), txt


def groups(model):
    """Layer list, confabulated and fact content tokens of one model."""
    d = json.load(open(ROOT / FILES[model]))
    layers = [str(x) for x in d["layers"]]
    items = {i["id"]: i for i in d["items"]}
    konf = []
    for k, r in KONF[model].items():
        p, txt = pos(items[k], r)
        if p is None:
            raise SystemExit(f"{model} {k}: content-token regex does not match: {txt[:120]!r}")
        konf.append((k, items[k]["tokens"][p]))
    fact = []
    for k, i in items.items():
        if i["kat"] == "fakt" and i["richtig"] and k not in FACT_EXCLUDED.get(model, ()):
            p, _ = pos(i)
            fact.append((k, i["tokens"][p]))
    return layers, konf, fact


def row(name, xs, layers, upper=4):
    none = int(layers[-1]) + 1
    h_all = [st.mean(t["layer"][x]["H"] for x in layers) for _, t in xs]
    h_up = [st.mean(t["layer"][x]["H"] for x in layers[-upper:]) for _, t in xs]
    lock = [t["kipp"] if t["kipp"] is not None else none for _, t in xs]
    p_out = [t["layer"]["out"]["top3"][0][1] for _, t in xs]
    span = f"L{layers[0]}-{layers[-1]}"
    up = f"L{layers[-upper]}-{layers[-1]}"
    print(f" {name:9} n={len(xs):2}  H({span})={st.mean(h_all):.2f}  H({up})={st.mean(h_up):.2f}  "
          f"lock-in median={st.median(lock)}  p_out={st.mean(p_out):.2f}")
    return h_up, p_out, lock


def main():
    for model in ("basis", "sft", "heretic"):
        layers, konf, fact = groups(model)
        print(f"== {model}")
        row("confab", konf, layers)
        row("confab∩4", [x for x in konf if x[0] in SHARED], layers)
        row("fact", fact, layers)
    layers, konf, fact = groups("gemma")
    print("== gemma (one confabulation only: a single case, not a group result)")
    row("confab", konf, layers)
    h_up, p_out, lock = row("fact", fact, layers)
    print(f"   fact range: H(upper) {min(h_up):.2f}–{max(h_up):.2f}, p_out min {min(p_out):.2f}, "
          f"lock-in max {max(lock)}")


if __name__ == "__main__":
    main()
