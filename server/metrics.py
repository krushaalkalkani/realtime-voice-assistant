"""Latency budget: turn the raw event timestamps into a per-turn breakdown.

Timeline of one turn (all times from the event log):

  user stops   VAD decides      ASR done   LLM starts  1st token   text to TTS   1st audio   speaker
  talking      (speech_end)    (asr_final) (llm_start) (llm_first) (tts_start)  (tts_first)  (playback_start)
     |--endpointing--|----asr-----|--------- llm_ttft -------|--llm_wait--|--tts_ttfb--|--delivery--|
                     |<------------------------- total (pipeline) ---------------------------------->|
     |<----------------------------- perceived (what the user feels) ------------------------------>|

  endpointing  silence the VAD waits for before deciding you're done (VAD_MIN_SILENCE_MS)
  asr          Whisper transcribing the utterance
  llm_ttft     LLM time-to-first-token (includes prompt processing / model load)
  llm_wait     first token -> first text we can speak (whole reply in "full" mode,
               first sentence in "sentence" mode)
  tts_ttfb     TTS time-to-first-byte of audio
  overhead     everything else: gaps between stages, sending audio, browser scheduling
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STAGES = ["endpointing_ms", "asr_ms", "llm_ttft_ms", "llm_wait_ms", "tts_ttfb_ms", "overhead_ms"]
STAGE_LABELS = {
    "endpointing_ms": "Endpointing (VAD silence)",
    "asr_ms": "ASR final",
    "llm_ttft_ms": "LLM TTFT",
    "llm_wait_ms": "LLM text wait",
    "tts_ttfb_ms": "TTS first byte",
    "overhead_ms": "Overhead",
    "total_ms": "Pipeline (speech end → audio)",
    "perceived_ms": "Total to first audio (user-perceived)",
}


def _t(events: dict[str, dict], name: str) -> float | None:
    e = events.get(name)
    return e["t_ms"] if e else None


def turn_breakdown(turn_events: list[dict]) -> dict | None:
    """turn_events: the event dicts of ONE turn. Returns None if the turn didn't reach playback."""
    first: dict[str, dict] = {}
    for e in turn_events:
        first.setdefault(e["type"], e)  # first occurrence of each event type

    # A degraded turn (filler, fallback model, text-only...) doesn't follow the normal
    # stage order, so only its end-to-end numbers are meaningful.
    if "fallback" in first:
        if "speech_end" not in first or "audio_playback_start" not in first:
            return None
        total = first["audio_playback_start"]["t_ms"] - first["speech_end"]["t_ms"]
        endp = float(first["speech_end"]["data"].get("trailing_silence_ms", 0))
        return {**{k: None for k in STAGES if k != "endpointing_ms"}, "endpointing_ms": endp,
                "total_ms": round(total, 1), "perceived_ms": round(total + endp, 1), "degraded": True}

    need = ["speech_end", "asr_final", "llm_start", "llm_first_token", "tts_start",
            "tts_first_byte", "audio_playback_start"]
    if any(n not in first for n in need):
        return None
    t = {n: _t(first, n) for n in need}

    b = {
        "endpointing_ms": float(first["speech_end"]["data"].get("trailing_silence_ms", 0)),
        "asr_ms": t["asr_final"] - t["speech_end"],
        "llm_ttft_ms": t["llm_first_token"] - t["llm_start"],
        "llm_wait_ms": t["tts_start"] - t["llm_first_token"],
        "tts_ttfb_ms": t["tts_first_byte"] - t["tts_start"],
        "total_ms": t["audio_playback_start"] - t["speech_end"],
    }
    b["overhead_ms"] = b["total_ms"] - (b["asr_ms"] + b["llm_ttft_ms"] + b["llm_wait_ms"] + b["tts_ttfb_ms"])
    b["perceived_ms"] = b["endpointing_ms"] + b["total_ms"]
    return {**{k: round(v, 1) for k, v in b.items()}, "degraded": False}


# ---------------------------------------------------------------- across sessions
def load_failures(sessions_dir: Path) -> list[dict]:
    """One row per stage failure (with what the pipeline did about it) across all sessions."""
    rows = []
    for path in sorted(Path(sessions_dir).glob("*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        start = next((e for e in events if e["type"] == "session_start"), None)
        label = (start or {}).get("data", {}).get("config", {}).get("label", "live")
        actions = {(e["turn_id"], e["data"]["stage"]): e["data"]["action"]
                   for e in events if e["type"] == "fallback"}
        for e in events:
            if e["type"] == "stage_failed":
                d = e["data"]
                rows.append({"session_id": path.stem, "label": label, "turn_id": e["turn_id"],
                             "stage": d["stage"], "reason": d["reason"], "detail": d.get("detail", ""),
                             "elapsed_ms": d.get("elapsed_ms", 0),
                             "action": actions.get((e["turn_id"], d["stage"]), "")})
            elif e["type"] == "barge_in":
                rows.append({"session_id": path.stem, "label": label, "turn_id": e["turn_id"],
                             "stage": "playback", "reason": "barge_in", "detail": "user talked over the coach",
                             "elapsed_ms": 0, "action": "stop_audio_and_listen"})
    return rows



def load_turns(sessions_dir: Path) -> list[dict]:
    """One row per completed turn across all session logs, with the session's config attached."""
    rows = []
    for path in sorted(Path(sessions_dir).glob("*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        start = next((e for e in events if e["type"] == "session_start"), None)
        config = (start or {}).get("data", {}).get("config", {})
        by_turn: dict[int, list[dict]] = {}
        for e in events:
            by_turn.setdefault(e["turn_id"], []).append(e)
        for turn_id, evs in sorted(by_turn.items()):
            b = turn_breakdown(evs)
            if b is None:
                continue
            asr = next((e for e in evs if e["type"] == "asr_final"), None)
            b = {k: (float("nan") if v is None else v) for k, v in b.items()}
            rows.append({
                "session_id": path.stem, "label": config.get("label", "live"), "turn_id": turn_id,
                "transcript": asr["data"].get("text", "") if asr else "", "config": config, **b,
            })
    return rows


def percentiles(rows: list[dict], keys=None) -> dict[str, dict[str, float]]:
    keys = keys or STAGES + ["total_ms", "perceived_ms"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]  # degraded turns have no per-stage numbers
        if len(vals):
            out[k] = {"p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
                      "n": int(len(vals))}
    return out
