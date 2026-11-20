"""Export recorded sessions + benchmark summaries as small JSON files for the showcase site.

Only BENCHMARK and FAULT-INJECTION sessions are exported. Their "user" is Piper-generated
speech, so no real person's voice or words leave the machine. Live sessions in logs/sessions/
are never read. Local file paths, configs and wall-clock timestamps are dropped; only
relative timings (ms since the turn started) are kept.

Usage:  uv run python scripts/export_site_data.py
Output: site/data/sessions.json, site/data/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from server.metrics import STAGE_LABELS, STAGES, load_turns, percentiles, turn_breakdown  # noqa: E402

BENCH = ROOT / "logs" / "bench"
FAULTS = ROOT / "logs" / "faults"
OUT = ROOT / "site" / "data"

DROP_KEYS = {"log", "config", "breakdown", "session_id", "ts"}
FAULT_NAMES = [  # order of scripts/fault_injection.py: 9 scenarios, barge-in, random run, real outage
    "ASR down", "ASR hangs", "LLM down", "LLM hangs (timeout)", "LLM slow (5 s)",
    "LLM stalls mid-reply", "TTS down", "TTS hangs", "LLM + TTS down",
    "Barge-in (user talks over the coach)", "Random faults, 8 turns",
    "Ollama unreachable (real outage)",
]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def clean(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in DROP_KEYS}


def export_session(path: Path, kind: str, title: str) -> dict:
    events = load(path)
    start = next(e for e in events if e["type"] == "session_start")
    cfg = start["data"].get("config", {})
    by_turn: dict[int, list[dict]] = {}
    for e in events:
        if e["type"] in ("session_start", "session_end"):
            continue
        by_turn.setdefault(e["turn_id"], []).append(e)

    turns = []
    for t, evs in sorted(by_turn.items()):
        t0 = evs[0]["t_ms"]
        first = {}
        for e in evs:
            first.setdefault(e["type"], e)
        user = (first.get("asr_final") or first.get("text_input") or {}).get("data", {}).get("text", "")
        dones = [e for e in evs if e["type"] == "llm_done"]
        reply = dones[-1]["data"].get("text", "") if dones else ""
        interrupted = first.get("turn_interrupted")
        if not reply and interrupted:
            reply = interrupted["data"].get("partial_reply", "")
        turns.append({
            "turn": t,
            "user": user,
            "typed": "text_input" in first,
            "reply": reply,
            "faults": (first.get("speech_end") or first.get("text_input") or {}).get("data", {}).get("faults") or {},
            "breakdown": turn_breakdown(evs),
            "fallbacks": [f"{e['data']['stage']}: {e['data']['action'].replace('_', ' ')}"
                          for e in evs if e["type"] == "fallback"],
            "interrupted": bool(interrupted),
            "events": [{"t": round(e["t_ms"] - t0, 1), "type": e["type"], "d": clean(e["data"])} for e in evs],
        })
    return {
        "id": path.stem, "kind": kind, "title": title,
        "config": {k: cfg.get(k) for k in ("asr_model", "llm_model", "tts_mode", "tts_fast_first", "warmup",
                                           "vad_min_silence_ms", "llm_fallback_model") if k in cfg},
        "turns": turns,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = []
    for label, title in (("9-final", "Benchmark: shipped config"), ("1-baseline", "Benchmark: Phase 1 baseline")):
        for p in sorted((BENCH / label).glob("*.jsonl")):
            sessions.append(export_session(p, "benchmark", title))
    fault_logs = sorted(FAULTS.glob("*.jsonl"))
    if len(fault_logs) != len(FAULT_NAMES):
        sys.exit(f"expected {len(FAULT_NAMES)} fault sessions, found {len(fault_logs)}")
    for p, name in zip(fault_logs, FAULT_NAMES):
        sessions.append(export_session(p, "fault", name))

    summary = {"stages": [{"key": k, "label": STAGE_LABELS[k]} for k in STAGES], "configs": []}
    for d in sorted(BENCH.iterdir()):
        rows = load_turns(d)
        if not rows:
            continue
        p = percentiles(rows)
        first = [r["perceived_ms"] for r in rows if r["turn_id"] == 1]
        summary["configs"].append({
            "label": d.name, "n": len(rows), "first_turn_ms": first[0] if first else None,
            **{k: {"p50": round(v["p50"]), "p95": round(v["p95"])} for k, v in p.items()},
        })

    (OUT / "sessions.json").write_text(json.dumps(sessions, ensure_ascii=False, separators=(",", ":")))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    blob = (OUT / "sessions.json").read_text()
    assert "/Users/" not in blob and "krushal" not in blob.lower(), "local path leaked into export"
    n_turns = sum(len(s["turns"]) for s in sessions)
    print(f"{len(sessions)} sessions, {n_turns} turns -> site/data/sessions.json "
          f"({len(blob) / 1024:.0f} KB); {len(summary['configs'])} configs -> site/data/summary.json")


if __name__ == "__main__":
    main()
