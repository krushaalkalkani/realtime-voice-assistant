"""Typed, timestamped events.

Every stage of the pipeline reports what it's doing by emitting an Event.
Each event goes to two places:
  1. a JSONL file per session (logs/sessions/<session_id>.jsonl) - for later analysis
  2. the browser over the WebSocket - so you can watch the pipeline live

Phase 2 will compute the latency budget purely from these events.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class EventType(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    # user side
    SPEECH_START = "speech_start"            # VAD heard the user start talking
    SPEECH_END = "speech_end"                # VAD decided the user finished
    ASR_PARTIAL = "asr_partial"              # rough transcript while user is still talking
    ASR_FINAL = "asr_final"                  # final transcript of the utterance
    # assistant side
    LLM_START = "llm_start"                  # request sent to the LLM
    LLM_FIRST_TOKEN = "llm_first_token"      # first token streamed back
    LLM_DONE = "llm_done"                    # full reply generated
    TTS_START = "tts_start"                  # text handed to TTS
    TTS_FIRST_BYTE = "tts_first_byte"        # first audio bytes produced
    TTS_DONE = "tts_done"                    # all audio produced and sent
    AUDIO_PLAYBACK_START = "audio_playback_start"  # browser started playing (reported by client)
    AUDIO_PLAYBACK_END = "audio_playback_end"      # browser finished playing (reported by client)
    TURN_COMPLETE = "turn_complete"
    TURN_SKIPPED = "turn_skipped"            # e.g. empty transcript, nothing to answer
    # resilience (Phase 3)
    TEXT_INPUT = "text_input"                # user typed instead of speaking
    STAGE_FAILED = "stage_failed"            # a stage timed out / errored  (data: stage, reason)
    FALLBACK = "fallback"                    # what we did instead          (data: stage, action)
    BARGE_IN = "barge_in"                    # user started talking over the coach
    TURN_INTERRUPTED = "turn_interrupted"    # the coach's turn was cut short
    ERROR = "error"                          # unexpected bug


class Event(BaseModel):
    type: EventType
    session_id: str
    turn_id: int
    ts: float = Field(description="wall-clock unix time (seconds)")
    t_ms: float = Field(description="milliseconds since session start (monotonic clock)")
    audio_pos: int = Field(0, description="mic samples received so far (aligns events with the recording)")
    data: dict[str, Any] = Field(default_factory=dict)


Sender = Callable[[dict], Awaitable[None]]


class EventBus:
    """Creates events, appends them to the session log, and forwards them to the client."""

    def __init__(self, session_id: str, log_dir: Path, send: Sender | None = None,
                 audio_pos: Callable[[], int] = lambda: 0):
        self.session_id = session_id
        self._audio_pos = audio_pos
        self._t0 = time.perf_counter()
        self._send = send
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{session_id}.jsonl"
        self._file = self.log_path.open("a", encoding="utf-8")
        self.history: list[Event] = []

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    async def emit(self, type: EventType, turn_id: int, **data: Any) -> Event:
        event = Event(
            type=type,
            session_id=self.session_id,
            turn_id=turn_id,
            ts=time.time(),
            t_ms=round(self.now_ms(), 1),
            audio_pos=self._audio_pos(),
            data=data,
        )
        self.history.append(event)
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()
        if self._send is not None:
            await self._send({"kind": "event", **json.loads(event.model_dump_json())})
        return event

    def find(self, type: EventType, turn_id: int) -> Event | None:
        """Most recent event of this type in this turn."""
        for e in reversed(self.history):
            if e.type == type and e.turn_id == turn_id:
                return e
        return None

    def close(self) -> None:
        self._file.close()
