"""Replay mode: re-run a recorded session through the pipeline, then diff against the original.

Every live session saves two files side by side (logs/sessions/ by default):
    <session_id>.jsonl   all events, each stamped with audio_pos (mic samples received so far)
    <session_id>.wav     the raw 16 kHz mic stream exactly as the server received it

Replay feeds the recorded mic audio frame by frame into a fresh Session and re-delivers the
browser's messages (playback_end, typed text) at the same audio positions. It runs on a
LOGICAL clock: each coach turn's server-side work finishes before more audio is fed, so the
result doesn't depend on how fast this machine is today. VAD and Whisper are deterministic,
so the same recording gives the same turns, transcripts and barge-ins every time.

By default the LLM replies come from the recording (fast, fully deterministic) and TTS is a
silent stub. Change one thing to see its effect, e.g.:

    uv run python replay/replay.py --list
    uv run python replay/replay.py <session_id>                          # should match exactly
    uv run python replay/replay.py <session_id> --set VAD_MIN_SILENCE_MS=400   # does it cut me off?
    uv run python replay/replay.py <session_id> --set ASR_MODEL=tiny.en       # transcripts change?
    uv run python replay/replay.py <session_id> --faults asr=down             # how would it degrade?
    uv run python replay/replay.py <session_id> --llm live --tts live         # full re-run
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import sys
import wave
from collections import deque
from pathlib import Path
from typing import AsyncIterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.asr.base import ASR  # noqa: E402
from server.config import Settings, settings  # noqa: E402
from server.faults import FaultInjector, FaultyASR, FaultyLLM, FaultyTTS  # noqa: E402
from server.llm.base import LLM  # noqa: E402
from server.pipeline import Session, State  # noqa: E402
from server.tts.base import TTS  # noqa: E402

SEARCH_DIRS = [ROOT / "logs" / "sessions", *sorted((ROOT / "logs").glob("bench/*")),
               ROOT / "logs" / "faults"]
REPLAY_DIR = ROOT / "logs" / "replays"
FRAME = 512


# ---------------------------------------------------------------- recorded stand-ins
class RecordedLLM(LLM):
    """Answers the n-th user message with the coach's reply to the n-th user message in the
    recording - and fails / is cold on exactly the turns where the real model was."""
    name = "recorded"
    model = "recorded"

    def __init__(self, turns: list[dict], role: str):
        self.turns, self.role = turns, role  # recorded user turns that reached the LLM, in order

    def _nth(self, messages: list[dict]) -> dict | None:
        n = sum(1 for m in messages if m["role"] == "user")
        return self.turns[n - 1] if 0 < n <= len(self.turns) else None

    async def is_warm(self) -> bool | None:
        rec = self._nth(self.history)
        return not (rec and rec["cold"])

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        rec = self._nth(messages)
        if rec and self.role in rec["llm_failed"] and not (self.role == "primary" and rec["faults"].get("llm")):
            raise ConnectionError("failed in the recording")
        for word in (rec["reply"] if rec else "Okay.").split(" "):
            yield word + " "


class RecordedBreaker:
    """Circuit-breaker decisions depend on how long earlier turns took, which a replay can't
    reproduce exactly - so they are replayed as recorded inputs too."""

    def __init__(self, name: str, open_turns: set[int], session_ref: list):
        self.name, self.open_turns, self.session_ref = name, open_turns, session_ref

    @property
    def is_open(self) -> bool:
        return self.session_ref[0].responding_turn in self.open_turns

    def record_failure(self): pass
    def record_success(self): pass


class SilentTTS(TTS):
    """Instant TTS stub: 50 ms of silence per text (we only care about the pipeline logic).
    Fails on the turns where the real TTS failed without an injected fault (e.g. starved of
    CPU while a model reloaded) - those failures are recorded inputs too."""
    name = "silent"
    sample_rate = 22050

    def __init__(self, fail_turns: set[int] = frozenset(), session_ref: list | None = None):
        self.fail_turns, self.session_ref = fail_turns, session_ref

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if self.session_ref and self.session_ref[0].responding_turn in self.fail_turns:
            raise ConnectionError("failed in the recording")
        yield b"\x00\x00" * (self.sample_rate // 20)


# ---------------------------------------------------------------- loading
def find_session(session_id: str) -> tuple[Path, Path]:
    for d in SEARCH_DIRS:
        log, wav = d / f"{session_id}.jsonl", d / f"{session_id}.wav"
        if log.exists() and wav.exists():
            return log, wav
    sys.exit(f"No recording found for {session_id} (need .jsonl + .wav). Try --list.")


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def summarize(events: list[dict]) -> list[dict]:
    """Per user turn: what the user said, what the coach answered, and what went wrong."""
    turns: dict[int, dict] = {}
    for e in events:
        t, d, typ = e["turn_id"], e["data"], e["type"]
        if t == 0:
            continue
        row = turns.setdefault(t, {"turn": t, "user": "", "reply": "", "flags": [], "faults": {},
                                   "start_pos": e["audio_pos"], "cold": False, "llm_failed": set(),
                                   "_role": None, "_token": set()})
        if typ == "speech_start":
            row["start_pos"] = e["audio_pos"]
        if typ == "llm_start":
            row["_role"] = d.get("role", "primary")
        if typ == "llm_first_token":
            row["_token"].add(row["_role"])
        if typ == "stage_failed" and d["stage"] == "llm" and d["reason"] != "circuit_open" \
                and row["_role"] not in row["_token"]:
            row["llm_failed"].add(row["_role"])
        if typ in ("speech_end", "text_input") and d.get("faults"):
            row["faults"] = d["faults"]
        if typ == "text_input":
            row["start_pos"] = e["audio_pos"]
        if typ in ("asr_final", "text_input"):
            row["user"] = d.get("text", "")
        elif typ == "llm_done":
            row["reply"] = d.get("text", "")
        elif typ == "turn_interrupted":
            row["reply"] = row["reply"] or d.get("partial_reply", "")
            row["flags"].append("interrupted")
        elif typ == "turn_skipped":
            row["flags"].append("skipped")
        elif typ == "fallback":
            row["flags"].append(f"{d['stage']}:{d['action']}")
            row["cold"] = row["cold"] or d["action"] == "filler_while_model_loads"
        if typ == "stage_failed" and d["reason"] == "circuit_open":
            row.setdefault("breaker_open", set()).add(d["stage"])
        if typ == "stage_failed" and d["stage"] == "tts" and d["reason"] != "circuit_open" \
                and not row["faults"].get("tts"):
            row["tts_env_failed"] = True
        elif typ == "speech_end" and d.get("discarded"):
            row["flags"].append("blip")
    out = []
    for r in sorted(turns.values(), key=lambda r: r["turn"]):
        if r["user"] or r["flags"]:
            out.append(r)
    return out


def comparable(r: dict) -> dict:
    # flags from concurrent tasks (speaker vs. LLM) may be logged in either order
    return {"user": r["user"], "reply": r["reply"], "flags": sorted(r["flags"])}


# ---------------------------------------------------------------- replay
async def replay(session_id: str, overrides: dict, faults: str, llm_mode: str, tts_mode: str,
                 asr_mode: str) -> tuple[list[dict], list[dict], Path]:
    log_path, wav_path = find_session(session_id)
    events = load_events(log_path)
    audio = load_audio(wav_path)
    rec_turns = summarize(events)

    # Start from the settings the session was RECORDED with, then apply --set overrides.
    start = next(e for e in events if e["type"] == "session_start")
    recorded_cfg = {k: v for k, v in start["data"].get("config", {}).items()
                    if k in {f.name for f in dataclasses.fields(Settings)} and k not in ("faults", "label")}
    s = dataclasses.replace(settings, **{
        **recorded_cfg,
        "sessions_dir": REPLAY_DIR, "run_label": f"replay:{session_id}", "record_audio": False,
        "asr_partial_every_ms": 0, "warmup": False,  # partials only affect the screen
        **overrides,
    })

    # components
    if asr_mode == "live":
        from server.asr import build_asr
        asr: ASR = build_asr(s)
    else:
        texts = deque(r["user"] for r in rec_turns if r["user"])

        class RecordedASR(ASR):
            name = "recorded"

            async def transcribe(self, a):
                return texts.popleft() if texts else ""
        asr = RecordedASR()
    if llm_mode == "live":
        from server.llm import build_llm
        llm: LLM = build_llm(s)
        fallback = build_llm(dataclasses.replace(s, llm_model=s.llm_fallback_model)) if s.llm_fallback_model else None
    else:
        # Failures that came from the environment (Ollama down, model too slow to load) are
        # inputs too: the recorded models fail on exactly the turns where the real ones did.
        llm_turns = [r for r in rec_turns if r["user"]]
        llm, fallback = RecordedLLM(llm_turns, "primary"), RecordedLLM(llm_turns, "fallback")
    if tts_mode == "live":
        from server.tts import build_tts
        tts: TTS = build_tts(s)
    else:
        session_ref: list = []
        tts = SilentTTS({r["turn"] for r in rec_turns if r.get("tts_env_failed")}, session_ref)
    inj = FaultInjector(faults)
    # Faults that were switched on during the recording are re-applied turn by turn,
    # at the audio position where that turn started (unless --faults overrides them).
    fault_schedule = deque() if faults else deque(sorted(
        (r["start_pos"], r["faults"]) for r in rec_turns))
    all_ok = {c: "ok" for c in inj.faults}

    async def send_json(m):
        pass

    async def send_bytes(b):
        pass

    # The breakers' clock follows the recording: at audio position p it reads the wall time the
    # live session had reached at p (time passes between turns even when no audio arrives).
    marks = sorted((e["audio_pos"], e["t_ms"] / 1000) for e in events)

    def clock() -> float:
        pos = session.audio_pos
        t = 0.0
        last_pos = 0
        for p, ts in marks:
            if p > pos:
                break
            t, last_pos = max(t, ts), p
        return t + (pos - last_pos) / 16000

    session = Session(s, FaultyASR(asr, inj), FaultyLLM(llm, inj), FaultyTTS(tts, inj),
                      send_json, send_bytes, llm_fallback=fallback, clock=clock)
    if isinstance(llm, RecordedLLM):
        llm.history = session.history
    if isinstance(tts, SilentTTS):
        session_ref.append(session)
    if not faults:  # replaying the recording as-is: breaker decisions come from the recording
        ref = [session]
        for name in session.breakers:
            session.breakers[name] = RecordedBreaker(
                name, {r["turn"] for r in rec_turns if name in r.get("breaker_open", set())}, ref)

    # browser messages to re-deliver, positioned on the recorded audio timeline
    typed = deque((e["audio_pos"], e["data"]["text"]) for e in events if e["type"] == "text_input")
    playback_ends = deque(sorted(e["audio_pos"] for e in events if e["type"] == "audio_playback_end"))

    async def settle(pos: int | None):
        """Finish the coach's server-side work, then play the browser's part."""
        task = session._turn_task
        if task and not task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if session.state == State.RESPONDING and session.responding_turn is not None:
            # the browser confirmed playback at a recorded position; deliver it when we get there
            if pos is None or not playback_ends or playback_ends[0] <= pos:
                if playback_ends:
                    playback_ends.popleft()
                t = session.responding_turn
                await session.handle_control({"type": "playback_start", "turn_id": t})
                await session.handle_control({"type": "playback_end", "turn_id": t})

    await session.start()
    await settle(0)
    for pos in range(0, len(audio) - FRAME + 1, FRAME):
        while fault_schedule and fault_schedule[0][0] <= pos + FRAME:
            inj.update({**all_ok, **fault_schedule.popleft()[1]})
        while typed and typed[0][0] <= pos:
            await session.handle_control({"type": "text_input", "text": typed.popleft()[1]})
        if session.state == State.RESPONDING:
            await settle(pos)
        await session.handle_audio(audio[pos:pos + FRAME].tobytes())
    await settle(None)
    await session.close()
    return rec_turns, summarize(load_events(session.bus.log_path)), session.bus.log_path


def short(text: str, n: int = 48) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", nargs="?")
    ap.add_argument("--list", action="store_true", help="list replayable sessions")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override settings for the replay, e.g. VAD_MIN_SILENCE_MS=400")
    ap.add_argument("--faults", default="", help='inject faults, e.g. "asr=down"')
    ap.add_argument("--asr", choices=["live", "recorded"], default="live")
    ap.add_argument("--llm", choices=["recorded", "live"], default="recorded")
    ap.add_argument("--tts", choices=["silent", "live"], default="silent")
    args = ap.parse_args()

    if args.list or not args.session_id:
        for d in SEARCH_DIRS:
            for wav in sorted(d.glob("*.wav")):
                n = sum(1 for e in load_events(wav.with_suffix(".jsonl")) if e["type"] in ("asr_final", "text_input"))
                print(f"{wav.stem}  {wav.stat().st_size / 32000:6.1f} s audio  {n:3d} user turns  ({d.relative_to(ROOT)})")
        return

    fields = {f.name: f for f in dataclasses.fields(Settings)}
    overrides = {}
    for kv in args.set:
        name, _, value = kv.partition("=")
        key = name.lower()
        if key not in fields:
            sys.exit(f"unknown setting {name}")
        default = getattr(settings, key)
        overrides[key] = (value.lower() in ("1", "true", "yes")) if isinstance(default, bool) else type(default)(value)

    rec, new, out = asyncio.run(replay(args.session_id, overrides, args.faults, args.llm, args.tts, args.asr))

    print(f"\nReplay of {args.session_id}  (asr={args.asr}, llm={args.llm}, tts={args.tts}"
          f"{', faults=' + args.faults if args.faults else ''}{', set ' + ' '.join(args.set) if args.set else ''})")
    print(f"{'#':>3}  {'user said (recorded → replay)':<52} {'coach':<7} flags")
    same = 0
    for i in range(max(len(rec), len(new))):
        a = rec[i] if i < len(rec) else {"user": "", "reply": "", "flags": ["(missing)"]}
        b = new[i] if i < len(new) else {"user": "", "reply": "", "flags": ["(missing)"]}
        user_ok, reply_ok = a["user"] == b["user"], a["reply"] == b["reply"]
        flags_ok = sorted(a["flags"]) == sorted(b["flags"])
        same += user_ok and reply_ok and flags_ok
        user = short(a["user"]) if user_ok else f"{short(a['user'], 24)} → {short(b['user'], 24)}"
        flags = ",".join(b["flags"]) or "-"
        if not flags_ok:
            flags = f"{','.join(a['flags']) or '-'} → {flags}"
        print(f"{i + 1:>3}  {'✓' if user_ok else '✗'} {user:<50} {'✓' if reply_ok else '✗':<7} {flags}")
    total = max(len(rec), len(new))
    print(f"\n{same}/{total} turns identical.  Replay log: {out.relative_to(ROOT)}")
    sys.exit(0 if same == total else 1)


if __name__ == "__main__":
    main()
