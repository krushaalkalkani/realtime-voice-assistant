"""Orchestration: one Session per connected browser tab.

Flow of one turn:
    mic frames -> TurnDetector (VAD) -> "end" -> ASR -> LLM (streamed) -> TTS -> audio to browser
                                                                        -> browser plays it
    browser says "playback ended" -> we start listening again.

TTS modes (settings.tts_mode):
  * "full":     wait for the whole LLM reply, then speak it          (Phase 1 behaviour)
  * "sentence": speak each sentence as soon as the LLM finishes it   (pipeline overlap, Phase 2)

Resilience (Phase 3) - every stage is time-limited and has a plan B:
  * ASR fails      -> say so, and offer the text box               (stage_failed + fallback events)
  * LLM too slow   -> say "give me a second", switch to the fallback model;
                      if that fails too -> a canned "could you say that again?"
  * LLM stalls     -> keep the part of the reply we already have
  * TTS fails      -> text-only reply (the text is already on screen)
  * circuit breaker: a component that just failed is skipped for a while instead of
                      making every turn wait for its timeout
  * browser never confirms playback -> watchdog resumes listening
  * barge-in       -> the user talks over the coach: stop audio, cancel the turn, listen
  * the mic audio is recorded next to the event log, so the session can be replayed
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
import wave
from datetime import datetime
from enum import StrEnum
from typing import Awaitable, Callable

import numpy as np

from server.asr.base import ASR
from server.chunker import SentenceChunker
from server.config import Settings
from server.events import EventBus, EventType as E
from server.llm.base import LLM
from server.metrics import turn_breakdown
from server.resilience import CircuitBreaker, StageFailure, stream_with_timeouts, with_timeout
from server.tts.base import TTS
from server.vad import SileroVAD, TurnDetector


class State(StrEnum):
    IDLE = "idle"              # connected, user hasn't pressed Start
    LISTENING = "listening"    # waiting for / capturing user speech
    RESPONDING = "responding"  # ASR -> LLM -> TTS -> playback in progress


def clean_for_speech(text: str) -> str:
    """Remove markdown-ish symbols the TTS would read out loud or choke on."""
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Session:
    def __init__(
        self,
        s: Settings,
        asr: ASR,
        llm: LLM,
        tts: TTS,
        send_json: Callable[[dict], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        llm_fallback: LLM | None = None,
        faults_state: Callable[[], dict] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.s = s
        self._faults_state = faults_state or dict
        self.asr, self.llm, self.tts, self.llm_fallback = asr, llm, tts, llm_fallback
        self.send_json, self.send_bytes = send_json, send_bytes

        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.audio_pos = 0  # mic samples received (and recorded) since Start
        self.bus = EventBus(self.id, s.sessions_dir, send=send_json, audio_pos=lambda: self.audio_pos)
        self.turns = TurnDetector(SileroVAD(frame_samples=s.frame_samples), s)
        # time source for the circuit breakers; replay passes the recording's timeline
        self.breakers = {n: CircuitBreaker(n, s.breaker_cooldown_s, clock or time.monotonic)
                         for n in ("asr", "llm", "tts")}

        self.state = State.IDLE
        self.turn_id = 0
        self.responding_turn: int | None = None  # the coach turn currently in progress
        self.history: list[dict] = [{"role": "system", "content": s.system_prompt}]

        self._pending = np.zeros(0, dtype=np.float32)  # mic samples not yet a full frame
        self._turn_task: asyncio.Task | None = None
        self._partial_task: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._last_partial_ms = 0.0
        self._reply_so_far: list[str] = []   # for history when a turn is interrupted
        self._recorder: wave.Wave_write | None = None

    def _active_faults(self) -> dict:
        """Injected faults in force right now - recorded per turn so replay can re-apply them."""
        return {k: v for k, v in self._faults_state().items() if v != "ok"}

    # ------------------------------------------------------------------ control
    async def set_state(self, state: State) -> None:
        self.state = state
        await self.send_json({"kind": "state", "state": state.value, "turn_id": self.turn_id})

    async def start(self) -> None:
        if self.state != State.IDLE:
            return
        if self.s.record_audio:
            self._recorder = wave.open(str(self.s.sessions_dir / f"{self.id}.wav"), "wb")
            self._recorder.setnchannels(1)
            self._recorder.setsampwidth(2)
            self._recorder.setframerate(self.s.sample_rate)
        await self.bus.emit(
            E.SESSION_START, 0,
            asr=self.asr.name, llm=f"{self.llm.name}:{getattr(self.llm, 'model', '')}",
            tts=self.tts.name, log=str(self.bus.log_path), config=self.s.snapshot(),
        )
        # Turn 0 = the coach's spoken greeting (TTS only, no ASR/LLM).
        self.history.append({"role": "assistant", "content": self.s.greeting})
        await self._begin_responding(0, self._speak_fixed(0, self.s.greeting))

    async def _begin_responding(self, turn_id: int, coro) -> None:
        self.responding_turn = turn_id
        self._reply_so_far = []
        await self.set_state(State.RESPONDING)
        self._turn_task = asyncio.create_task(coro)

    async def handle_control(self, msg: dict) -> None:
        kind = msg.get("type")
        turn_id = msg.get("turn_id", self.turn_id)
        if kind == "start":
            await self.start()
        elif kind == "playback_start":
            await self.bus.emit(E.AUDIO_PLAYBACK_START, turn_id, source="client")
        elif kind == "playback_end":
            await self._on_playback_end(turn_id)
        elif kind == "text_input":
            await self._on_text_input(str(msg.get("text", "")).strip())

    async def _on_playback_end(self, turn_id: int) -> None:
        await self.bus.emit(E.AUDIO_PLAYBACK_END, turn_id, source="client")
        if self.state != State.RESPONDING or turn_id != self.responding_turn:
            return  # stale message from an interrupted turn
        await self._complete_turn(turn_id)

    async def _complete_turn(self, turn_id: int) -> None:
        if self._watchdog and self._watchdog is not asyncio.current_task():
            self._watchdog.cancel()
        turn_events = [e.model_dump(mode="json") for e in self.bus.history if e.turn_id == turn_id]
        breakdown = turn_breakdown(turn_events)  # None for the greeting (no user speech)
        degraded = [f"{e.data['stage']}:{e.data['action']}" for e in self.bus.history
                    if e.turn_id == turn_id and e.type == E.FALLBACK]
        await self.bus.emit(E.TURN_COMPLETE, turn_id, breakdown=breakdown,
                            target_ms=self.s.latency_target_ms, degraded=degraded)
        await self._resume_listening()

    async def _resume_listening(self) -> None:
        self.responding_turn = None
        self.turns.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        await self.set_state(State.LISTENING)

    async def _on_text_input(self, text: str) -> None:
        if not text or self.state == State.IDLE:
            return
        if self.state == State.RESPONDING:
            await self._interrupt("text_input")
        self.turn_id += 1
        await self.bus.emit(E.TEXT_INPUT, self.turn_id, text=text, faults=self._active_faults())
        self.turns.reset()
        await self._begin_responding(self.turn_id, self._run_turn(self.turn_id, text=text))

    # ------------------------------------------------------------------ barge-in
    async def _interrupt(self, reason: str) -> None:
        """Cut the coach off: cancel generation/TTS and tell the browser to stop playing."""
        old = self.responding_turn
        for task in (self._turn_task, self._watchdog):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        await self.send_json({"kind": "stop_audio", "turn_id": old})
        partial = "".join(self._reply_so_far).strip()
        await self.bus.emit(E.TURN_INTERRUPTED, old if old is not None else self.turn_id,
                            reason=reason, partial_reply=partial)
        self.responding_turn = None
        self.state = State.LISTENING  # note: the VAD is NOT reset - the user is mid-sentence

    # ------------------------------------------------------------------ audio in
    async def handle_audio(self, data: bytes) -> None:
        if self.state == State.IDLE:
            return
        if self._recorder:
            self._recorder.writeframes(data)
        self.audio_pos += len(data) // 2
        if self.state == State.RESPONDING and not self.s.barge_in:
            return  # half-duplex: ignore the mic while the coach talks
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        n = self.s.frame_samples
        while len(self._pending) >= n and self.state != State.IDLE:
            frame, self._pending = self._pending[:n], self._pending[n:]
            await self._process_frame(frame)

    async def _process_frame(self, frame: np.ndarray) -> None:
        result = self.turns.process(frame)

        if self.state == State.RESPONDING:
            # Only listening for barge-in: is the user clearly talking over the coach?
            if result in ("end", "discard"):
                self.turns.take_utterance()  # a short blip (or echo); ignore it
            elif self.turns.speaking and self.turns.speech_frames_ms >= self.s.barge_in_min_speech_ms:
                await self.bus.emit(E.BARGE_IN, self.responding_turn or 0,
                                    speech_ms=round(self.turns.speech_frames_ms))
                await self._interrupt("barge_in")
                self.turn_id += 1
                self._last_partial_ms = 0.0
                await self.bus.emit(E.SPEECH_START, self.turn_id, barge_in=True)
                await self.set_state(State.LISTENING)
            return

        if result == "start":
            self.turn_id += 1
            self._last_partial_ms = 0.0
            await self.bus.emit(E.SPEECH_START, self.turn_id, prob=round(self.turns.last_prob, 3))
        elif result == "discard":
            await self.bus.emit(E.SPEECH_END, self.turn_id, discarded=True, reason="too short")
        elif result == "end":
            audio = self.turns.take_utterance()
            await self.bus.emit(
                E.SPEECH_END, self.turn_id,
                audio_ms=round(len(audio) / self.s.sample_rate * 1000),
                trailing_silence_ms=round(self.turns.trailing_silence_ms),
                faults=self._active_faults(),
            )
            await self._begin_responding(self.turn_id, self._run_turn(self.turn_id, audio=audio))
        elif self.turns.speaking:
            self._maybe_partial()

    def _maybe_partial(self) -> None:
        every = self.s.asr_partial_every_ms
        if every <= 0 or self.breakers["asr"].is_open:
            return
        if self._partial_task and not self._partial_task.done():
            return
        if self.turns.speech_ms - self._last_partial_ms < every:
            return
        self._last_partial_ms = self.turns.speech_ms
        audio, turn_id = self.turns.current_audio(), self.turn_id

        async def run():
            with contextlib.suppress(Exception):  # partials are best-effort
                text = await asyncio.wait_for(self.asr.transcribe(audio), self.s.asr_timeout_s)
                # drop stale partials that finish after the user already stopped talking
                if text and self.state == State.LISTENING and self.turn_id == turn_id:
                    await self.bus.emit(E.ASR_PARTIAL, turn_id, text=text)

        self._partial_task = asyncio.create_task(run())

    # ------------------------------------------------------------------ one turn
    async def _run_turn(self, turn_id: int, audio: np.ndarray | None = None, text: str | None = None):
        to_speak: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speaker(turn_id, to_speak))
        try:
            # 1) speech -> text (skipped when the user typed)
            if text is None:
                text = await self._transcribe(turn_id, audio)
                if text is None:  # ASR failed: apologise and point to the text box
                    await self.send_json({"kind": "asr_unavailable", "turn_id": turn_id})
                    await self._say(to_speak, turn_id, self.s.asr_failed_text)
                    await to_speak.put(None)
                    await speaker
                    return
                if not text.strip():
                    speaker.cancel()
                    await self.bus.emit(E.TURN_SKIPPED, turn_id, reason="empty transcript")
                    await self._resume_listening()
                    return
            self.history.append({"role": "user", "content": text})

            # 2) text -> reply, streamed into the speaker as sentences complete
            reply = await self._generate(turn_id, to_speak)
            self.history.append({"role": "assistant", "content": reply})
            await to_speak.put(None)
            await speaker  # 3) finishes when all audio is sent (or TTS gave up)
        except asyncio.CancelledError:
            speaker.cancel()
            partial = "".join(self._reply_so_far).strip()
            if partial:  # barge-in: remember what the coach managed to say
                self.history.append({"role": "assistant", "content": partial + " ..."})
            raise
        except Exception as e:  # a bug, not a component failure: log it, keep the session alive
            speaker.cancel()
            await self.bus.emit(E.ERROR, turn_id, stage="turn", error=f"{type(e).__name__}: {e}")
            await self._resume_listening()

    async def _transcribe(self, turn_id: int, audio: np.ndarray) -> str | None:
        """ASR with a time limit. Returns None if ASR is unavailable."""
        breaker = self.breakers["asr"]
        if breaker.is_open:
            await self._fail(turn_id, StageFailure("asr", "circuit_open", "skipping: failed recently"),
                             action="ask_to_type")
            return None
        timeout = self.s.asr_timeout_s + self.s.asr_timeout_per_audio_s * len(audio) / self.s.sample_rate
        try:
            text = await with_timeout("asr", self.asr.transcribe(audio), timeout)
        except StageFailure as f:
            breaker.record_failure()
            await self._fail(turn_id, f, action="ask_to_type")
            return None
        breaker.record_success()
        await self.bus.emit(E.ASR_FINAL, turn_id, text=text)
        return text

    async def _generate(self, turn_id: int, to_speak: asyncio.Queue) -> str:
        """Get a reply: primary LLM -> (filler + fallback LLM) -> canned line. Never hangs."""
        attempts = [("primary", self.llm, self.s.llm_first_token_timeout_s)]
        if self.llm_fallback is not None:
            attempts.append(("fallback", self.llm_fallback, self.s.llm_fallback_timeout_s))

        for i, (role, llm, first_timeout) in enumerate(attempts):
            more = i + 1 < len(attempts)
            if role == "primary" and self.breakers["llm"].is_open:
                await self._fail(turn_id, StageFailure("llm", "circuit_open", "primary failed recently"),
                                 action="use_fallback_model" if more else "canned_reply")
                continue
            if role == "primary" and await llm.is_warm() is False:
                # The model was evicted from memory (e.g. Ollama swapped in the fallback model).
                # Reloading takes seconds: say so now and allow for it, instead of timing out
                # and falling back again (which would evict it again - thrashing).
                first_timeout = self.s.llm_cold_timeout_s
                await self.bus.emit(E.FALLBACK, turn_id, stage="llm", action="filler_while_model_loads")
                await self.send_json({"kind": "assistant_note", "turn_id": turn_id, "text": self.s.filler_text})
                await to_speak.put(self.s.filler_text)
            try:
                reply = await self._stream_reply(turn_id, llm, role, first_timeout, to_speak)
                if role == "primary":
                    self.breakers["llm"].record_success()
                return reply
            except StageFailure as f:
                if role == "primary":
                    self.breakers["llm"].record_failure()
                model = getattr(llm, "model", "")
                if f.partial:  # it stalled mid-reply: keep what we have, it's already on screen
                    await self._fail(turn_id, f, action="keep_partial_reply", model=model)
                    await self.bus.emit(E.LLM_DONE, turn_id, text=f.partial, partial=True, model=model)
                    return f.partial
                await self._fail(turn_id, f, model=model,
                                 action="filler_then_fallback_model" if more else "canned_reply")
                if more and role == "primary" and first_timeout != self.s.llm_cold_timeout_s:
                    await self.send_json({"kind": "assistant_note", "turn_id": turn_id,
                                          "text": self.s.filler_text})
                    await to_speak.put(self.s.filler_text)  # spoken right away, while we retry

        reply = self.s.llm_failed_text
        await self.bus.emit(E.LLM_DONE, turn_id, text=reply, canned=True)
        await self._say(to_speak, turn_id, reply)
        return reply

    async def _stream_reply(self, turn_id: int, llm: LLM, role: str, first_timeout: float,
                            to_speak: asyncio.Queue) -> str:
        sentence_mode = self.s.tts_mode == "sentence"
        chunker = SentenceChunker(
            self.s.sentence_min_chars,
            first_clause_min_chars=self.s.fast_first_min_chars if self.s.tts_fast_first else None,
        )
        model = getattr(llm, "model", "")
        await self.bus.emit(E.LLM_START, turn_id, model=model, role=role)
        self._reply_so_far = []
        n_tokens = 0
        try:
            async for piece in stream_with_timeouts("llm", llm.stream(self.history),
                                                    first_timeout, self.s.llm_stall_timeout_s):
                if n_tokens == 0:
                    await self.bus.emit(E.LLM_FIRST_TOKEN, turn_id, model=model)
                n_tokens += 1
                self._reply_so_far.append(piece)
                await self.send_json({"kind": "llm_delta", "turn_id": turn_id, "text": piece})
                if sentence_mode:
                    for sentence in chunker.push(piece):
                        await to_speak.put(sentence)
        except StageFailure as f:
            spoken = "".join(self._reply_so_far).strip()
            if spoken:
                for rest in (chunker.flush() if sentence_mode else [spoken]):
                    await to_speak.put(rest)
            f.partial = spoken
            raise
        reply = "".join(self._reply_so_far).strip()
        await self.bus.emit(E.LLM_DONE, turn_id, tokens=n_tokens, text=reply, model=model)
        for rest in (chunker.flush() if sentence_mode else [reply]):
            await to_speak.put(rest)
        return reply

    async def _say(self, to_speak: asyncio.Queue, turn_id: int, text: str) -> None:
        """Speak a canned line and also show it (in case TTS is down too)."""
        await self.send_json({"kind": "assistant_text", "turn_id": turn_id, "text": text})
        await to_speak.put(text)

    async def _speak_fixed(self, turn_id: int, text: str) -> None:
        """Speak a fixed text (the greeting)."""
        await self.send_json({"kind": "assistant_text", "turn_id": turn_id, "text": text})
        q: asyncio.Queue = asyncio.Queue()
        await q.put(text)
        await q.put(None)
        await self._speaker(turn_id, q)

    async def _speaker(self, turn_id: int, to_speak: asyncio.Queue) -> None:
        """Synthesize each text from the queue and stream the audio to the browser.
        If TTS fails, the rest of the turn is text-only. Always ends with TTS_DONE so the
        browser (or the watchdog) can finish the turn."""
        total_bytes, n_texts, tts_ok = 0, 0, True
        breaker = self.breakers["tts"]
        while (text := await to_speak.get()) is not None:
            speech = clean_for_speech(text)
            if not speech or not tts_ok:
                continue
            if breaker.is_open:
                tts_ok = False
                await self.send_json({"kind": "tts_unavailable", "turn_id": turn_id})
                await self._fail(turn_id, StageFailure("tts", "circuit_open", "failed recently"),
                                 action="text_only")
                continue
            if n_texts == 0:
                await self.bus.emit(E.TTS_START, turn_id, sample_rate=self.tts.sample_rate, chars=len(speech))
            n_texts += 1
            try:
                async for chunk in stream_with_timeouts("tts", self.tts.stream(speech),
                                                        self.s.tts_timeout_s, self.s.tts_timeout_s):
                    if total_bytes == 0:
                        await self.bus.emit(E.TTS_FIRST_BYTE, turn_id)
                    total_bytes += len(chunk)
                    await self.send_bytes(chunk)
                breaker.record_success()
            except StageFailure as f:
                breaker.record_failure()
                tts_ok = False
                await self.send_json({"kind": "tts_unavailable", "turn_id": turn_id})
                await self._fail(turn_id, f, action="text_only")
        if n_texts == 0:  # nothing spoken; still let the browser finish the turn
            await self.bus.emit(E.TTS_START, turn_id, sample_rate=self.tts.sample_rate, chars=0)
        audio_ms = total_bytes / 2 / self.tts.sample_rate * 1000
        await self.bus.emit(E.TTS_DONE, turn_id, bytes=total_bytes, audio_ms=round(audio_ms), texts=n_texts)
        # If the browser never reports playback_end (tab closed, audio blocked...),
        # don't stay stuck in RESPONDING forever.
        self._watchdog = asyncio.create_task(self._playback_watchdog(turn_id, audio_ms))

    async def _playback_watchdog(self, turn_id: int, audio_ms: float) -> None:
        await asyncio.sleep(audio_ms / 1000 + self.s.playback_grace_s)
        if self.state == State.RESPONDING and self.responding_turn == turn_id:
            await self._fail(turn_id, StageFailure("playback", "timeout", "browser never confirmed playback"),
                             action="resume_listening")
            await self._complete_turn(turn_id)

    async def _fail(self, turn_id: int, f: StageFailure, action: str, **extra) -> None:
        """Record a failure and what we did about it - this is what makes failures visible."""
        await self.bus.emit(E.STAGE_FAILED, turn_id, stage=f.stage, reason=f.reason,
                            detail=f.detail, elapsed_ms=f.elapsed_ms, **extra)
        await self.bus.emit(E.FALLBACK, turn_id, stage=f.stage, action=action)

    # ------------------------------------------------------------------ teardown
    async def close(self) -> None:
        for task in (self._turn_task, self._partial_task, self._watchdog):
            if task and not task.done():
                task.cancel()
        try:
            await self.bus.emit(E.SESSION_END, self.turn_id)
        except Exception:
            pass  # socket already closed; the event is still written to the log file
        if self._recorder:
            self._recorder.close()
        self.bus.close()
