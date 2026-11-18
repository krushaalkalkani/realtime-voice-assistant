"""Full pipeline with REAL VAD but FAKE ASR/LLM/TTS: checks the orchestration and event order."""

import asyncio
import dataclasses

import numpy as np

from server.asr.base import ASR
from server.config import settings
from server.llm.base import LLM
from server.pipeline import Session, State
from server.tts.base import TTS
from tests.conftest import frames


class FakeASR(ASR):
    name = "fake"
    async def transcribe(self, audio):
        return "I build voice apps."


class FakeLLM(LLM):
    name = "fake"
    async def stream(self, messages):
        for tok in ["Great ", "answer. ", "Next question?"]:
            yield tok


class FakeTTS(TTS):
    name = "fake"
    sample_rate = 16_000
    def __init__(self):
        self.texts = []
    async def stream(self, text):
        self.texts.append(text)
        yield b"\x00\x00" * 1600
        yield b"\x00\x00" * 1600


def run_session(speech_16k, tmp_path, tts_mode):
    s = dataclasses.replace(settings, sessions_dir=tmp_path, asr_partial_every_ms=0, tts_mode=tts_mode, tts_fast_first=False)
    tts = FakeTTS()
    sent_json, sent_bytes = [], []

    async def send_json(m): sent_json.append(m)
    async def send_bytes(b): sent_bytes.append(b)

    async def scenario():
        sess = Session(s, FakeASR(), FakeLLM(), tts, send_json, send_bytes)
        await sess.handle_control({"type": "start"})
        await sess._turn_task                      # greeting spoken (turn 0)
        await sess.handle_control({"type": "playback_end", "turn_id": 0})
        assert sess.state == State.LISTENING

        silence = np.zeros(16_000, dtype=np.float32)
        audio = np.concatenate([silence, speech_16k, silence, silence])
        for f in frames(audio):
            await sess.handle_audio((f * 32767).astype("<i2").tobytes())
        await sess._turn_task                      # ASR -> LLM -> TTS for turn 1
        await sess.handle_control({"type": "playback_start", "turn_id": 1})
        await sess.handle_control({"type": "playback_end", "turn_id": 1})
        await sess.close()
        return sess

    return asyncio.run(scenario()), tts, sent_bytes


def test_one_full_turn(speech_16k, tmp_path):
    sess, tts, sent_bytes = run_session(speech_16k, tmp_path, "full")
    turn1 = [e.type.value for e in sess.bus.history if e.turn_id == 1 and e.type != "session_end"]
    assert turn1 == [
        "speech_start", "speech_end", "asr_final", "llm_start", "llm_first_token", "llm_done",
        "tts_start", "tts_first_byte", "tts_done", "audio_playback_start", "audio_playback_end",
        "turn_complete",
    ]
    assert sess.history[-2:] == [
        {"role": "user", "content": "I build voice apps."},
        {"role": "assistant", "content": "Great answer. Next question?"},
    ]
    assert len(sent_bytes) == 4  # 2 greeting chunks + 2 reply chunks
    done = sess.bus.find("turn_complete", 1)
    assert done.data["breakdown"]["endpointing_ms"] > 600   # VAD waited ~700 ms of silence
    assert done.data["breakdown"]["total_ms"] >= 0
    assert sess.bus.log_path.read_text().count("\n") == len(sess.bus.history)


def test_sentence_mode_speaks_each_sentence_separately(speech_16k, tmp_path):
    sess, tts, _ = run_session(speech_16k, tmp_path, "sentence")
    assert tts.texts[1:] == ["Great answer.", "Next question?"]  # [0] is the greeting
    assert sess.bus.find("tts_done", 1).data["texts"] == 2
