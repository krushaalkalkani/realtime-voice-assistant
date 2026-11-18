import asyncio

from server.asr import build_asr
from server.config import settings


def test_whisper_transcribes_piper_speech(speech_16k):
    asr = build_asr(settings)
    text = asyncio.run(asr.transcribe(speech_16k)).lower()
    assert "software engineer" in text
