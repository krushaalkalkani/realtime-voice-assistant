"""A headless stand-in for the browser, used by the smoke test, benchmark and fault injection.

Like web/app.js it: streams "mic" audio at real-time pace, receives the coach's audio,
reports playback_start ~30 ms after the first audio chunk, reports playback_end when the
turn's audio is done, and stops playing when the server says stop_audio (barge-in).
A background task reads server messages continuously, so we can talk while the coach talks.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.config import settings  # noqa: E402
from tests.conftest import frames, resample  # noqa: E402

BROWSER_SCHEDULE_S = 0.03  # app.js schedules the first chunk 30 ms ahead

_voice = None


def speak(text: str) -> np.ndarray:
    """Synthesize `text` with Piper -> 16 kHz float32 (our fake 'user voice')."""
    global _voice
    from piper import PiperVoice

    if _voice is None:
        _voice = PiperVoice.load(settings.models_dir / "piper" / f"{settings.piper_voice}.onnx")
    audio = np.concatenate([c.audio_float_array for c in _voice.synthesize(text)])
    return resample(audio, _voice.config.sample_rate)


class FakeBrowser:
    def __init__(self, url: str, verbose: bool = True, realtime_playback: bool = False):
        self.url, self.verbose = url, verbose
        # realtime_playback: wait as long as the audio lasts before reporting playback_end
        # (needed to talk over the coach); otherwise report it as soon as all audio arrived.
        self.realtime_playback = realtime_playback
        self.events: list[dict] = []
        self.messages: list[dict] = []
        self.audio: dict[int, bytearray] = {}
        self.replies: dict[int, str] = {}
        self.transcripts: dict[int, str] = {}
        self.done: dict[int, str] = {}  # turn_id -> "ok" | "interrupted" | "skipped" | "error"
        self.sample_rate = 22050
        self._turn = 0
        self._accepting = True
        self._play_started: dict[int, float] = {}

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        self._reader = asyncio.create_task(self._read_loop())
        await self._send({"type": "start"})
        return self

    async def __aexit__(self, *exc):
        self._reader.cancel()
        await self.ws.close()

    async def _send(self, msg: dict, delay: float = 0.0):
        if delay:
            await asyncio.sleep(delay)
        await self.ws.send(json.dumps(msg))

    # ------------------------------------------------------------ incoming
    async def _read_loop(self):
        async for msg in self.ws:
            if isinstance(msg, bytes):
                self._on_audio(msg)
            else:
                await self._on_json(json.loads(msg))

    def _on_audio(self, data: bytes):
        if not self._accepting:
            return
        t = self._turn
        self.audio.setdefault(t, bytearray()).extend(data)
        if t not in self._play_started:  # first audio of this turn -> "playback starts"
            self._play_started[t] = time.perf_counter() + BROWSER_SCHEDULE_S
            asyncio.create_task(self._send({"type": "playback_start", "turn_id": t}, BROWSER_SCHEDULE_S))

    async def _on_json(self, m: dict):
        self.messages.append(m)
        kind = m["kind"]
        if kind in ("llm_delta", "assistant_text"):
            self.replies[m["turn_id"]] = self.replies.get(m["turn_id"], "") + m["text"]
        if kind == "stop_audio":
            self._accepting = False
        if kind != "event":
            return
        self.events.append(m)
        t, d, typ = m["turn_id"], m["data"], m["type"]
        if self.verbose:
            shown = {k: v for k, v in d.items() if k not in ("text", "config", "breakdown", "partial_reply")}
            print(f"  {m['t_ms']:>9.1f} ms  #{t} {typ:<20} {json.dumps(shown)}")
        if typ in ("asr_final", "text_input"):
            self.transcripts[t] = d["text"]
        elif typ == "tts_start":
            self.sample_rate, self._turn, self._accepting = d["sample_rate"], t, True
        elif typ == "tts_done":
            asyncio.create_task(self._finish_playback(t))
        elif typ == "turn_interrupted":
            self.done[t] = "interrupted"
        elif typ == "turn_skipped":
            self.done[t] = "skipped"
        elif typ == "error":
            self.done[t] = "error"

    async def _finish_playback(self, t: int):
        await asyncio.sleep(BROWSER_SCHEDULE_S + 0.01)  # playback_start goes out first
        if self.realtime_playback and t in self._play_started:
            audio_s = len(self.audio.get(t, b"")) / 2 / self.sample_rate
            await asyncio.sleep(max(0.0, self._play_started[t] + audio_s - time.perf_counter()))
        if t in self.done:  # interrupted meanwhile
            return
        await self._send({"type": "playback_end", "turn_id": t})
        self.done[t] = "ok"

    # ------------------------------------------------------------ waiting helpers
    async def wait_turn(self, turn_id: int, timeout: float = 60) -> bool:
        """Wait until the coach's turn `turn_id` is finished. True if it completed normally."""
        deadline = time.perf_counter() + timeout
        while turn_id not in self.done:
            if time.perf_counter() > deadline:
                raise TimeoutError(f"turn {turn_id} did not finish within {timeout}s")
            await asyncio.sleep(0.01)
        return self.done[turn_id] == "ok"

    async def wait_event(self, typ: str, turn_id: int | None = None, timeout: float = 60) -> dict:
        deadline = time.perf_counter() + timeout
        while True:
            for e in self.events:
                if e["type"] == typ and (turn_id is None or e["turn_id"] == turn_id):
                    return e
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no {typ} event within {timeout}s")
            await asyncio.sleep(0.01)

    def turn_events(self, turn_id: int) -> list[dict]:
        return [e for e in self.events if e["turn_id"] == turn_id]

    # ------------------------------------------------------------ outgoing
    async def turn(self, turn_id: int, audio: np.ndarray, timeout: float = 60) -> bool:
        """Speak one answer and wait for the coach to finish replying."""
        await self.say(audio)
        return await self.wait_turn(turn_id, timeout)

    async def say(self, audio: np.ndarray, lead_s: float = 0.5, tail_s: float = 2.0) -> None:
        """Stream audio (16 kHz float32) like a microphone, with silence before and after."""
        await asyncio.sleep(0.3)  # server switches back to listening after playback_end
        lead, tail = np.zeros(int(16000 * lead_s), np.float32), np.zeros(int(16000 * tail_s), np.float32)
        for f in frames(np.concatenate([lead, audio, tail])):
            await self.ws.send((f * 32767).astype("<i2").tobytes())
            await asyncio.sleep(0.032)  # real-time pace

    async def type_text(self, text: str) -> None:
        await self._send({"type": "text_input", "text": text})
