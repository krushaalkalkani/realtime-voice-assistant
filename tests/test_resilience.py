"""Phase 3: every failure mode degrades instead of hanging, and shows up in the events.

Fake components + the real fault injector + tiny timeouts, so each test runs in well under a second.
"""

import asyncio
import dataclasses
import time
import wave

import numpy as np
import pytest

from server.asr.base import ASR
from server.config import settings
from server.faults import FaultInjector, FaultyASR, FaultyLLM, FaultyTTS
from server.llm.base import LLM
from server.pipeline import Session, State
from server.tts.base import TTS
from tests.conftest import frames

FAST = dict(
    asr_timeout_s=0.3, asr_timeout_per_audio_s=0.0, llm_first_token_timeout_s=0.3,
    llm_stall_timeout_s=0.3, llm_fallback_timeout_s=0.5, tts_timeout_s=0.3, playback_grace_s=0.2,
    breaker_cooldown_s=60, asr_partial_every_ms=0, tts_mode="sentence", tts_fast_first=False,
)


class FakeASR(ASR):
    name = "fake"
    async def transcribe(self, audio):
        return "I build voice apps."


class FakeLLM(LLM):
    name = "fake"
    def __init__(self, words=("Great ", "answer. ", "Next ", "question?"), model="primary"):
        self.words, self.model = words, model
    async def stream(self, messages):
        for w in self.words:
            await asyncio.sleep(0)
            yield w


class FakeTTS(TTS):
    name = "fake"
    sample_rate = 16_000
    def __init__(self):
        self.texts = []
    async def stream(self, text):
        self.texts.append(text)
        yield b"\x00\x00" * 160


class Harness:
    def __init__(self, tmp_path, faults="", fallback=True, **overrides):
        self.s = dataclasses.replace(settings, sessions_dir=tmp_path, **{**FAST, **overrides})
        self.inj = FaultInjector(faults)
        self.tts = FakeTTS()
        self.json, self.bytes = [], []
        self.fallback = FakeLLM(("Fallback ", "reply. ", "Next?"), model="fallback") if fallback else None

    async def __aenter__(self):
        async def sj(m): self.json.append(m)
        async def sb(b): self.bytes.append(b)
        self.sess = Session(self.s, FaultyASR(FakeASR(), self.inj), FaultyLLM(FakeLLM(), self.inj),
                            FaultyTTS(self.tts, self.inj), sj, sb, llm_fallback=self.fallback)
        await self.sess.start()
        await self.sess._turn_task
        await self.sess.handle_control({"type": "playback_end", "turn_id": 0})
        return self

    async def __aexit__(self, *exc):
        await self.sess.close()

    async def typed_turn(self, text="hello"):
        await self.sess.handle_control({"type": "text_input", "text": text})
        await self.sess._turn_task
        return self.sess.turn_id

    async def finish(self, turn_id):
        await self.sess.handle_control({"type": "playback_end", "turn_id": turn_id})

    def events(self, turn_id, typ=None):
        return [e for e in self.sess.bus.history if e.turn_id == turn_id and (typ is None or e.type == typ)]

    def actions(self, turn_id):
        return [e.data["action"] for e in self.events(turn_id, "fallback")]


def run(coro):
    return asyncio.run(coro)


def test_llm_hang_speaks_filler_then_uses_fallback_model(tmp_path):
    async def go():
        async with Harness(tmp_path, faults="llm=hang") as h:
            t = await h.typed_turn()
            failed = h.events(t, "stage_failed")[0].data
            assert (failed["stage"], failed["reason"]) == ("llm", "timeout")
            assert h.actions(t) == ["filler_then_fallback_model"]
            assert h.tts.texts[1:] == ["Give me a second.", "Fallback reply.", "Next?"]
            assert h.sess.history[-1] == {"role": "assistant", "content": "Fallback reply. Next?"}
    run(go())


def test_llm_down_without_fallback_gives_canned_reply(tmp_path):
    async def go():
        async with Harness(tmp_path, faults="llm=down", fallback=False) as h:
            t = await h.typed_turn()
            assert h.actions(t) == ["canned_reply"]
            assert h.tts.texts[-1] == settings.llm_failed_text
    run(go())


def test_llm_stall_keeps_partial_reply(tmp_path):
    async def go():
        h = Harness(tmp_path, faults="llm=stall")
        async with h:
            h.sess.llm.inner.words = [f"w{i} " for i in range(10)] + ["end."]
            t = await h.typed_turn()
            assert h.actions(t) == ["keep_partial_reply"]
            assert h.sess.history[-1]["content"] == "w0 w1 w2 w3 w4 w5"
    run(go())


def test_tts_down_goes_text_only_and_breaker_skips_it_next_turn(tmp_path):
    async def go():
        async with Harness(tmp_path, faults="tts=down") as h:
            t1 = await h.typed_turn()
            assert h.actions(t1) == ["text_only"]
            assert h.events(t1, "tts_done")[0].data["bytes"] == 0
            assert any(m["kind"] == "tts_unavailable" for m in h.json)
            await h.finish(t1)

            h.inj.update("tts=hang")  # would cost a full timeout if we tried it again
            t0 = time.perf_counter()
            t2 = await h.typed_turn()
            assert time.perf_counter() - t0 < 0.2
            assert h.events(t2, "stage_failed")[0].data["reason"] == "circuit_open"
    run(go())


def test_asr_hang_times_out_and_asks_user_to_type(tmp_path):
    async def go():
        async with Harness(tmp_path, faults="asr=hang") as h:
            h.sess.turn_id = 1
            await h.sess._begin_responding(1, h.sess._run_turn(1, audio=np.zeros(16000, np.float32)))
            await h.sess._turn_task
            assert h.events(1, "stage_failed")[0].data["reason"] == "timeout"
            assert h.actions(1) == ["ask_to_type"]
            assert any(m["kind"] == "asr_unavailable" for m in h.json)
            assert h.tts.texts[-1] == settings.asr_failed_text
    run(go())


def test_missing_playback_confirmation_triggers_watchdog(tmp_path):
    async def go():
        async with Harness(tmp_path) as h:
            t = await h.typed_turn()
            assert h.sess.state == State.RESPONDING
            await asyncio.sleep(0.4)  # browser never sends playback_end
            assert h.sess.state == State.LISTENING
            assert h.actions(t) == ["resume_listening"]
    run(go())


def test_barge_in_during_playback_stops_audio_and_starts_new_turn(speech_16k, tmp_path):
    async def go():
        h = Harness(tmp_path)
        async with h:
            t = await h.typed_turn()  # coach is "talking" (no playback_end yet)
            audio = np.concatenate([speech_16k, np.zeros(16000, np.float32)])
            for f in frames(audio):
                await h.sess.handle_audio((f * 32767).astype("<i2").tobytes())
                if h.sess.state == State.RESPONDING and h.sess._turn_task.done() is False:
                    await h.sess._turn_task
            assert h.events(t, "barge_in")
            assert h.events(t, "turn_interrupted")[0].data["reason"] == "barge_in"
            assert any(m["kind"] == "stop_audio" for m in h.json)
            new = [e for e in h.sess.bus.history if e.type == "speech_start" and e.data.get("barge_in")]
            assert new and new[0].turn_id == t + 1
            await h.sess._turn_task  # the interrupting utterance becomes a normal turn
            assert h.events(t + 1, "asr_final")
    run(go())


def test_typing_while_coach_talks_interrupts(tmp_path):
    async def go():
        async with Harness(tmp_path, faults="llm=slow:200") as h:
            await h.sess.handle_control({"type": "text_input", "text": "first"})
            first = h.sess.turn_id
            await asyncio.sleep(0.05)
            t = await h.typed_turn("second")
            assert h.events(first, "turn_interrupted")[0].data["reason"] == "text_input"
            assert h.events(t, "llm_done")
    run(go())


def test_session_records_mic_audio_aligned_with_events(speech_16k, tmp_path):
    async def go():
        async with Harness(tmp_path) as h:
            for f in frames(np.concatenate([speech_16k, np.zeros(24000, np.float32)])):
                await h.sess.handle_audio((f * 32767).astype("<i2").tobytes())
            end = [e for e in h.sess.bus.history if e.type == "speech_end"][0]
            return h.sess, end
    sess, end = run(go())
    with wave.open(str(tmp_path / f"{sess.id}.wav")) as w:
        assert w.getframerate() == 16000
        assert w.getnframes() == sess.audio_pos
    assert 0 < end.audio_pos <= sess.audio_pos


def test_fault_spec_parsing():
    inj = FaultInjector("asr=down, llm=slow:250,tts=flaky:0.2")
    assert inj.describe() == {"asr": "down", "llm": "slow:250", "tts": "flaky:0.2"}
    with pytest.raises(ValueError):
        inj.update("llm=explode")


def test_cold_primary_model_gets_filler_and_longer_timeout_instead_of_fallback(tmp_path):
    class ColdLLM(FakeLLM):
        async def is_warm(self):
            return False
        async def stream(self, messages):
            await asyncio.sleep(0.5)  # "loading": longer than the normal 0.3 s first-token limit
            async for w in super().stream(messages):
                yield w

    async def go():
        h = Harness(tmp_path, llm_cold_timeout_s=2.0)
        async with h:
            h.sess.llm.inner = ColdLLM()
            t = await h.typed_turn()
            assert h.actions(t) == ["filler_while_model_loads"]
            assert not h.events(t, "stage_failed")              # no timeout, no fallback model
            assert h.tts.texts[1:] == ["Give me a second.", "Great answer.", "Next question?"]
    run(go())
