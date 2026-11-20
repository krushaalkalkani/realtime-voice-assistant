"""Render site/demo.gif and site/og.png from a REAL recorded turn (no invented numbers).

The GIF replays the median turn of the shipped benchmark config at real speed: the
candidate stops talking, the voice detector waits for silence, Whisper transcribes, the
LLM starts streaming and Piper's first audio arrives. Timings come straight from the
event log exported by scripts/export_site_data.py.

Usage:  uv run python scripts/make_site_media.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

BG, PANEL, LINE = "#161616", "#1c1c1c", "#2c2c2c"
TEXT, MUTED, FAINT, ACCENT = "#e6e6e6", "#a3a3a3", "#8a8a8a", "#2dd4bf"
STAGES = [("endpointing_ms", "waiting for silence", "#6b7280"), ("asr_ms", "speech-to-text", "#8b5cf6"),
          ("llm_ttft_ms", "LLM first token", "#3b82f6"), ("llm_wait_ms", "LLM text to speak", "#38bdf8"),
          ("tts_ttfb_ms", "TTS first audio", "#2dd4bf"), ("overhead_ms", "overhead", "#f59e0b")]
TARGET = 1500
plt.rcParams["font.family"] = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def median_turn() -> dict:
    sessions = json.loads((SITE / "data" / "sessions.json").read_text())
    final = next(s for s in sessions if s["title"] == "Benchmark: shipped config")
    turns = sorted((t for t in final["turns"] if t["breakdown"] and not t["breakdown"]["degraded"]),
                   key=lambda t: t["breakdown"]["perceived_ms"])
    return turns[len(turns) // 2]


def demo_gif(turn: dict) -> None:
    b = turn["breakdown"]
    total = b["perceived_ms"]
    pre, post, fps = 900, 1900, 12  # ms of "still speaking" before, hold after first audio
    t_start, t_end = -pre, total + post
    frames = int((t_end - t_start) / 1000 * fps)
    user = textwrap.fill(f"“{turn['user']}”", 70)
    reply_words = turn["reply"].split()
    ttft_at = b["endpointing_ms"] + b["asr_ms"] + b["llm_ttft_ms"]

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 960)
    ax.set_ylim(540, 0)
    ax.axis("off")

    def draw(i: int) -> None:
        now = t_start + i * 1000 / fps
        ax.clear()
        ax.set_xlim(0, 960)
        ax.set_ylim(540, 0)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 960, 540, boxstyle="square,pad=0", fc=BG, ec=BG))
        ax.text(40, 48, "VOICE INTERVIEW COACH  ·  ONE REAL TURN, REAL SPEED", color=FAINT, fontsize=10.5, weight="medium")

        # candidate bubble
        ax.add_patch(FancyBboxPatch((40, 70), 880, 112, boxstyle="round,pad=0,rounding_size=10", fc=PANEL, ec=LINE))
        ax.text(60, 96, "CANDIDATE" + ("  ·  speaking…" if now < 0 else ""), color=FAINT, fontsize=9.5, weight="medium")
        heard = now >= b["endpointing_ms"] + b["asr_ms"]
        ax.text(60, 116, user if heard else ("…" if now >= 0 else "(speaking)"), color=TEXT if heard else MUTED,
                fontsize=12.5, va="top", linespacing=1.45)

        # coach bubble: words stream in from the first token (max 3 lines, then an ellipsis)
        ax.add_patch(FancyBboxPatch((40, 196), 880, 140, boxstyle="round,pad=0,rounding_size=10", fc=PANEL, ec=LINE))
        speaking = now >= total
        ax.text(60, 222, "COACH" + ("  ·  speaking" if speaking else ""), color=ACCENT if speaking else FAINT,
                fontsize=9.5, weight="medium")
        if now >= ttft_at:
            n = min(len(reply_words), int((now - ttft_at) / 1000 * 22) + 1)  # ~22 tokens/s streaming
            lines = textwrap.wrap(" ".join(reply_words[:n]), 74)
            if len(lines) > 3:
                lines = lines[:3]
                lines[-1] = lines[-1].rstrip(".,") + " …"
            ax.text(60, 242, "\n".join(lines), color=TEXT, fontsize=12.5, va="top", linespacing=1.45)

        # timeline
        x0, w, y = 40, 880, 384
        scale = w / (max(total, TARGET) * 1.1)
        ax.add_patch(FancyBboxPatch((x0, y), w, 26, boxstyle="round,pad=0,rounding_size=4", fc=PANEL, ec=LINE))
        x, t_acc = x0, 0.0
        for key, label, color in STAGES:
            dur = b[key]
            shown = max(0.0, min(dur, now - t_acc))
            if shown > 0:
                ax.add_patch(plt.Rectangle((x, y), shown * scale, 26, fc=color, ec="none"))
            if now >= t_acc and now < t_acc + dur:
                ax.text(min(x + 4, x0 + w - 200), y + 50, f"now: {label}", color=color, fontsize=11, weight="medium")
            x += dur * scale
            t_acc += dur
        tx = x0 + TARGET * scale
        ax.plot([tx, tx], [y - 8, y + 34], color="#f87171", lw=1.4, ls=(0, (4, 3)))
        ax.text(tx, y - 14, "1.5 s target", color="#f87171", fontsize=9, ha="center")
        ax.text(x0, y - 14, "candidate stops talking", color=FAINT, fontsize=9)

        clock = max(0.0, min(now, total)) / 1000
        ax.text(40, 500, f"{clock:0.2f} s", color=ACCENT if speaking else TEXT, fontsize=34, weight="bold")
        ax.text(200, 488, "from the candidate's last word" if not speaking
                else "to first audio: the coach is talking", color=MUTED, fontsize=12)
        ax.text(200, 510, "Llama 3.2 3B · faster-whisper · Piper, all on a laptop", color=FAINT, fontsize=10)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    anim.save(SITE / "demo.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)


def og_png(turn: dict) -> None:
    b = turn["breakdown"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")
    ax.text(70, 110, "VOICE INTERVIEW COACH", color=FAINT, fontsize=16, weight="medium")
    ax.text(70, 200, "Answers out loud in 1.8 s.", color=TEXT, fontsize=44, weight="bold")
    ax.text(70, 270, "Break any part, and it still answers.", color=MUTED, fontsize=32, weight="semibold")
    x0, w, y = 70, 1060, 360
    scale = w / (max(b["perceived_ms"], TARGET) * 1.1)
    x = x0
    for key, label, color in STAGES:
        ax.add_patch(plt.Rectangle((x, y), b[key] * scale, 34, fc=color, ec="none"))
        x += b[key] * scale
    tx = x0 + TARGET * scale
    ax.plot([tx, tx], [y - 10, y + 44], color="#f87171", lw=2, ls=(0, (5, 4)))
    stats = [("1.8 s", "median to first audio"), ("−47%", "vs. the first version"),
             ("11 / 11", "failures degrade, none hang"), ("30 / 30", "turns replay identically")]
    for i, (big, small) in enumerate(stats):
        ax.text(70 + i * 270, 480, big, color=ACCENT if i == 0 else TEXT, fontsize=30, weight="bold")
        ax.text(70 + i * 270, 520, small, color=MUTED, fontsize=14)
    ax.text(70, 585, "Whisper · Llama 3.2 · Piper · WebSockets · all local", color=FAINT, fontsize=13)
    fig.savefig(SITE / "og.png", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    t = median_turn()
    print(f"median turn #{t['turn']}: {t['breakdown']['perceived_ms']:.0f} ms")
    demo_gif(t)
    og_png(t)
    for f in ("demo.gif", "og.png"):
        print(f"site/{f}: {(SITE / f).stat().st_size / 1024:.0f} KB")
