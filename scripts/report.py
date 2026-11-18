"""Turns the benchmark sessions (logs/bench/<label>/) into tables and charts for the README.

Outputs:
    docs/latency_results.md      p50/p95 per stage, before/after optimization table
    docs/latency_breakdown.png   stacked p50 bars, one per config, with the target line
    docs/waterfall.png           waterfall of a median turn of the final config

Usage:  uv run python scripts/report.py
"""

from __future__ import annotations

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import ALL_RUNS, ANSWERS, BENCH_DIR, FINAL  # noqa: E402
from fake_browser import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))
from server.config import settings  # noqa: E402
from server.metrics import STAGE_LABELS, STAGES, load_turns, percentiles  # noqa: E402

DOCS = ROOT / "docs"
COLORS = ["#9aa3b2", "#7c5cff", "#2f6fed", "#3fb4e8", "#1a9b5c", "#d98a00"]
TARGET = settings.latency_target_ms


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main():
    DOCS.mkdir(exist_ok=True)
    results = []  # (label, desc, rows, pct)
    for label, desc, _ in ALL_RUNS:
        rows = load_turns(BENCH_DIR / label)
        if rows:
            results.append((label, desc, rows, percentiles(rows)))
    if not results:
        sys.exit("No benchmark sessions found. Run scripts/benchmark.py first.")

    final_label, _, final_rows, final_p = next((r for r in results if r[0] == FINAL[0]), results[-1])
    md = [f"# Latency results\n\nBenchmark: the same {len(ANSWERS)} spoken answers replayed into every config "
          f"(cold start each run). Target: **< {TARGET} ms** from the user stopping talking to the first audio.\n"]

    # --- per-stage table for the final config
    md.append(f"## Stage breakdown — final config (`{final_label}`, n={len(final_rows)} turns)\n")
    md.append("| Stage | p50 (ms) | p95 (ms) |\n|---|---|---|")
    for k in STAGES + ["total_ms", "perceived_ms"]:
        name = STAGE_LABELS[k]
        if k == "perceived_ms":
            name = f"**{name}**"
        md.append(f"| {name} | {final_p[k]['p50']:.0f} | {final_p[k]['p95']:.0f} |")
    hit = final_p["perceived_ms"]["p50"] < TARGET
    md.append(f"\nTarget {TARGET} ms at p50: **{'met ✅' if hit else 'missed ❌'}** "
              f"(p95 {final_p['perceived_ms']['p95']:.0f} ms).\n")

    # --- before/after table
    md.append("## Optimizations (rows 1-8 are cumulative: each adds one change to the row above; "
              "`9-final` is the shipped combination, measured on its own, Δ vs baseline)\n")
    md.append("| Config | Change | Perceived p50 | p95 | Δ p50 vs previous | First turn | ASR WER |")
    md.append("|---|---|---|---|---|---|---|")
    prev = None
    for label, desc, rows, p in results:
        cur = p["perceived_ms"]["p50"]
        base = results[0][3]["perceived_ms"]["p50"]
        delta = "—" if prev is None else f"{cur - (base if label == FINAL[0] else prev):+.0f} ms"
        first = [r["perceived_ms"] for r in rows if r["turn_id"] == 1]
        w = np.mean([wer(ANSWERS[(r["turn_id"] - 1) % len(ANSWERS)], r["transcript"]) for r in rows])
        md.append(f"| `{label}` | {desc} | {cur:.0f} | {p['perceived_ms']['p95']:.0f} | {delta} | "
                  f"{first[0]:.0f} | {w:.1%} |" if first else f"| `{label}` | {desc} | {cur:.0f} | | {delta} | | |")
        prev = cur

    md.append("\n## Per-stage p50 for every config (ms)\n")
    md.append("| Config | " + " | ".join(STAGE_LABELS[k] for k in STAGES) + " | Perceived |")
    md.append("|---|" + "---|" * (len(STAGES) + 1))
    for label, _, _, p in results:
        md.append(f"| `{label}` | " + " | ".join(f"{p[k]['p50']:.0f}" for k in STAGES)
                  + f" | {p['perceived_ms']['p50']:.0f} |")
    (DOCS / "latency_results.md").write_text("\n".join(md) + "\n")

    # --- chart 1: stacked p50 bars per config
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(results) + 1.6))
    labels = [f"{lbl}" for lbl, *_ in results][::-1]
    left = np.zeros(len(results))
    for k, color in zip(STAGES, COLORS):
        vals = np.array([p[k]["p50"] for *_, p in results][::-1])
        ax.barh(labels, vals, left=left, color=color, label=STAGE_LABELS[k])
        left += vals
    ax.axvline(TARGET, color="#d23c3c", ls="--", lw=1.5, label=f"target {TARGET} ms")
    for y, total in enumerate(left):
        ax.text(total + 30, y, f"{total:.0f} ms", va="center", fontsize=9)
    ax.set_xlabel("ms from user stops talking to first audio (p50 per stage)")
    ax.set_title("Where the time goes — p50 of each stage, stacked (sum shown at the end)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, max(left.max(), TARGET) * 1.15)
    fig.tight_layout()
    fig.savefig(DOCS / "latency_breakdown.png", dpi=140)

    # --- chart 2: waterfall of the median turn in the final config
    med = sorted(final_rows, key=lambda r: r["perceived_ms"])[len(final_rows) // 2]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    start = 0.0
    names = [STAGE_LABELS[k] for k in STAGES][::-1]
    for i, (k, color) in enumerate(zip(STAGES, COLORS)):
        y = len(STAGES) - 1 - i
        ax.barh(y, med[k], left=start, color=color)
        ax.text(start + med[k] + 15, y, f"{med[k]:.0f}", va="center", fontsize=8)
        start += med[k]
    ax.set_yticks(range(len(STAGES)), names)
    ax.axvline(TARGET, color="#d23c3c", ls="--", lw=1.2)
    ax.set_xlabel("ms after the user stops talking")
    ax.set_title(f"Waterfall of a median turn ({final_label}): {med['perceived_ms']:.0f} ms to first audio")
    fig.tight_layout()
    fig.savefig(DOCS / "waterfall.png", dpi=140)

    print((DOCS / "latency_results.md").read_text())
    print(f"Charts: {DOCS}/latency_breakdown.png, {DOCS}/waterfall.png")


if __name__ == "__main__":
    main()
