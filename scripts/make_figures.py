"""All figures, in English and German. No GPU, no model weights.

Numbers come only through the analysis scripts (metrics, analyze_content_tokens,
analyze_h4) — this file computes nothing of its own.
Output: out/figures/{en,de}/<name>.{png,svg}

    python scripts/make_figures.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_content_tokens as act  # noqa: E402
import analyze_h4  # noqa: E402
import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/figures"

# Colour-blind-checked categorical palette, slots in fixed order
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"sft": "#2a78d6", "heretic": "#eb6834", "basis": "#1baf7a", "gemma": "#eda100"}
FACT, CONFAB = "#2a78d6", "#eb6834"
HARMFUL, HARMLESS = "#eb6834", "#2a78d6"
REPO = {"sft": "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B",
        "heretic": "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic",
        "basis": "Qwen/Qwen3.5-9B", "gemma": "google/gemma-4-12B-it"}

TEXTS = {
    "en": {
        "model": {"sft": "Qwen SFT", "heretic": "Heretic", "basis": "Qwen3.5-9B base", "gemma": "Gemma 4 12B"},
        "cat": {"fiktiv": "Fabricated item recognised as unknown", "fakt": "Facts correct", "mathe": "Math correct"},
        "scores_title": "Fact battery: answers after manual review",
        "scores_sub": "37 questions, greedy, no thinking · share per category",
        "layer": "Layer", "entropy": "Lens entropy (nats)",
        "fact": "fact, correct", "confab": "confabulated",
        "ent_title": "Lens entropy at the content token: invented vs. known",
        "ent_sub": "Mean over hand-marked content tokens · Qwen lens L20–30",
        "gem_title": "Gemma 4 12B: its only confabulation against all facts",
        "gem_sub": "Lens L33–46 · band = range of the 15 facts · n=1, not a group result",
        "gem_confab": "f09 “Hugo Wolf” (confabulated)", "gem_fact": "facts, mean", "gem_band": "facts, range",
        "h4_title": "Qwen3.5-9B, Heretic − SFT: lens entropy of the first k answer tokens",
        "h4_sub": "100 prompts each from Heretic's eval sets · Δ in nats",
        "h4_y": "Δ entropy (nats)", "harmful": "harmful_behaviors", "harmless": "harmless_alpaca",
        "ref_title": "Refusal rate (keyword rule, 32 tokens)",
        "ref_sub": "100 prompts each · harmful_behaviors and harmless_alpaca",
        "ref_y": "refused (%)",
    },
    "de": {
        "model": {"sft": "Qwen-SFT", "heretic": "Heretic", "basis": "Qwen3.5-9B Basis", "gemma": "Gemma 4 12B"},
        "cat": {"fiktiv": "Erfundenes als unbekannt erkannt", "fakt": "Fakten richtig", "mathe": "Mathe richtig"},
        "scores_title": "Faktenbatterie: Antworten nach Handprüfung",
        "scores_sub": "37 Fragen, greedy, ohne Thinking · Anteil je Kategorie",
        "layer": "Layer", "entropy": "Lens-Entropie (nats)",
        "fact": "Fakt, richtig", "confab": "konfabuliert",
        "ent_title": "Lens-Entropie am Inhaltstoken: erfunden vs. gewusst",
        "ent_sub": "Mittel über handmarkierte Inhaltstokens · Qwen-Lens L20–30",
        "gem_title": "Gemma 4 12B: der einzige konfabulierte Fall gegen alle Fakten",
        "gem_sub": "Lens L33–46 · Band = Spanne der 15 Fakten · n=1, kein Gruppenbeleg",
        "gem_confab": "f09 „Hugo Wolf“ (konfabuliert)", "gem_fact": "Fakten, Mittel", "gem_band": "Fakten, Spanne",
        "h4_title": "Qwen3.5-9B, Heretic − SFT: Lens-Entropie der ersten k Antworttokens",
        "h4_sub": "Je 100 Prompts aus Heretics Eval-Sets · Δ in nats",
        "h4_y": "Δ Entropie (nats)", "harmful": "harmful_behaviors", "harmless": "harmless_alpaca",
        "ref_title": "Ablehnungsquote (Keyword-Regel, 32 Tokens)",
        "ref_sub": "Je 100 Prompts · harmful_behaviors bzw. harmless_alpaca",
        "ref_y": "abgelehnt (%)",
    },
}


def num(lang, x, digits=2):
    s = f"{x:.{digits}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    s = s.replace("-", "−")
    return s.replace(".", ",") if lang == "de" else s


def pct(lang, x):
    return f"{x:.0f} %" if lang == "de" else f"{x:.0f}%"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def header(fig, title, sub):
    fig.text(0.02, 0.97, title, fontsize=13, fontweight="bold", va="top", color=TEXT)
    fig.text(0.02, 0.915, sub, fontsize=9.5, va="top", color=TEXT2)


def save(fig, lang, name, models):
    lines = [f"{TEXTS[lang]['model'][m]} = {REPO[m]}" for m in models]
    fig.text(0.02, 0.012, "\n".join(lines), fontsize=7.5, color=TEXT2, va="bottom", linespacing=1.4)
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.045 * len(lines) + 0.02)
    d = OUT / lang
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(d / f"{name}.{ext}", dpi=160, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def end_label(ax, lang, x, y, color):
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=2)
    ax.annotate(num(lang, y), (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=TEXT)


def scores(lang, items, hand):
    t = TEXTS[lang]
    models = ["gemma", "basis", "sft", "heretic"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, cat in zip(axs, ("fiktiv", "fakt", "mathe")):
        for y, m in enumerate(models):
            ids = [k for k, i in items[m].items() if i["kat"] == cat]
            n = sum(hand[m][k] == metrics.GOOD[cat] for k in ids)
            ax.barh(y, n / len(ids) * 100, height=0.5, color=MODEL_COLOR[m])
            ax.text(n / len(ids) * 100 + 2, y, f"{n}/{len(ids)}", va="center", fontsize=9, color=TEXT)
        ax.set_title(t["cat"][cat], fontsize=10, color=TEXT, loc="left")
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pct(lang, x)))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(range(len(models)), [t["model"][m] for m in models])
    axs[0].invert_yaxis()
    header(fig, t["scores_title"], t["scores_sub"])
    fig.subplots_adjust(left=0.14, right=0.98, top=0.76, bottom=0.12, wspace=0.12)
    save(fig, lang, "scores", models)


def profile(tokens, layers):
    return [st.mean(t["layer"][x]["H"] for _, t in tokens) for x in layers]


def entropy_layers(lang):
    t = TEXTS[lang]
    models = ("sft", "heretic", "basis")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, m in zip(axs, models):
        layers, konf, fact = act.groups(m)
        x = [int(a) for a in layers]
        for tokens, color, label in ((fact, FACT, t["fact"]), (konf, CONFAB, t["confab"])):
            y = profile(tokens, layers)
            ax.plot(x, y, color=color, label=f"{label} (n={len(tokens)})")
            end_label(ax, lang, x[-1], y[-1], color)
        ax.set_title(t["model"][m], fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel(t["layer"])
        ax.set_xticks(x[::2])
        ax.set_xlim(x[0] - 0.3, x[-1] + 1.6)
        ax.legend(loc="lower left", fontsize=8.5, handlelength=1.4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: num(lang, y, 0)))
    axs[0].set_ylabel(t["entropy"])
    axs[0].set_ylim(0, None)
    header(fig, t["ent_title"], t["ent_sub"])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.08)
    save(fig, lang, "entropy_layers", models)


def gemma(lang):
    t = TEXTS[lang]
    layers, konf, fact = act.groups("gemma")
    x = [int(a) for a in layers]
    lo = [min(tk["layer"][a]["H"] for _, tk in fact) for a in layers]
    hi = [max(tk["layer"][a]["H"] for _, tk in fact) for a in layers]
    mean = profile(fact, layers)
    confab = profile(konf, layers)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x, lo, hi, color=FACT, alpha=0.10, lw=0, label=t["gem_band"])
    ax.plot(x, mean, color=FACT, label=t["gem_fact"])
    ax.plot(x, confab, color=CONFAB, label=t["gem_confab"])
    end_label(ax, lang, x[-1], mean[-1], FACT)
    end_label(ax, lang, x[-1], confab[-1], CONFAB)
    ax.set_xlabel(t["layer"])
    ax.set_ylabel(t["entropy"])
    ax.set_xticks(x[::2])
    ax.set_xlim(x[0] - 0.3, x[-1] + 1.6)
    ax.set_ylim(0, 11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: num(lang, y, 0)))
    ax.legend(loc="upper left", fontsize=8.5, ncols=3, bbox_to_anchor=(0, 1.0))
    header(fig, t["gem_title"], t["gem_sub"])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
    save(fig, lang, "entropy_gemma", ("gemma",))


def h4(lang):
    t = TEXTS[lang]
    ks = (1, 3, 5)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (name, layers) in zip(axs, analyze_h4.RANGES.items()):
        for j, (g, color) in enumerate((("harmful", HARMFUL), ("harmless", HARMLESS))):
            d = [analyze_h4.value("heretic", g, k, layers) - analyze_h4.value("sft", g, k, layers) for k in ks]
            xs = [i + (j - 0.5) * 0.3 for i in range(len(ks))]
            ax.bar(xs, d, width=0.27, color=color, label=t[g])
            for xx, dd in zip(xs, d):
                sign = "+" if float(f"{dd:.2f}") > 0 else ""
                ax.text(xx, dd + (0.03 if dd >= 0 else -0.03), sign + num(lang, dd),
                        ha="center", va="bottom" if dd >= 0 else "top", fontsize=8, color=TEXT)
        ax.axhline(0, color=TEXT2, lw=1)
        ax.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
        ax.set_title(name, fontsize=10, color=TEXT, loc="left")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: num(lang, y, 1)))
    axs[0].set_ylabel(t["h4_y"])
    axs[0].set_ylim(-0.8, 1.1)
    axs[0].set_yticks([-0.5, 0, 0.5, 1.0])
    axs[1].legend(loc="upper right", fontsize=8.5)
    header(fig, t["h4_title"], t["h4_sub"])
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.1, wspace=0.08)
    save(fig, lang, "h4_delta", ("sft", "heretic"))


def refusals(lang):
    t = TEXTS[lang]
    models = ("sft", "heretic", "basis")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, (g, color) in enumerate((("harmful", HARMFUL), ("harmless", HARMLESS))):
        for i, m in enumerate(models):
            rows = json.load(open(metrics.SIGNALS[m]))["daten"][g]["zeilen"]
            q = sum(r["abgelehnt"] for r in rows) / len(rows) * 100
            xx = i + (j - 0.5) * 0.3
            ax.bar(xx, q, width=0.27, color=color, label=t[g] if i == 0 else None)
            ax.text(xx, q + 1.5, pct(lang, q), ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(range(len(models)), [t["model"][m] for m in models])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(t["ref_y"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center", fontsize=8.5, ncols=1, bbox_to_anchor=(0.5, 0.55))
    header(fig, t["ref_title"], t["ref_sub"])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.1)
    save(fig, lang, "refusals", models)


def main():
    style()
    items, hand = metrics.load()
    for lang in ("en", "de"):
        scores(lang, items, hand)
        entropy_layers(lang)
        gemma(lang)
        h4(lang)
        refusals(lang)
    print("\n".join(sorted(str(p.relative_to(ROOT)) for p in OUT.rglob("*.png"))))


if __name__ == "__main__":
    main()
