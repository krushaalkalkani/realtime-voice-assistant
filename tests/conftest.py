"""Shared test fixtures. We generate real speech with Piper so VAD/ASR tests use real audio."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.config import settings  # noqa: E402


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = 16_000) -> np.ndarray:
    n = int(len(audio) * dst_rate / src_rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


@pytest.fixture(scope="session")
def speech_16k() -> np.ndarray:
    """~2-3 s of synthetic speech at 16 kHz float32."""
    from piper import PiperVoice

    voice = PiperVoice.load(settings.models_dir / "piper" / f"{settings.piper_voice}.onnx")
    chunks = [c.audio_float_array for c in voice.synthesize("I am a software engineer who loves building voice apps.")]
    return resample(np.concatenate(chunks), voice.config.sample_rate)


def frames(audio: np.ndarray, n: int = 512):
    for i in range(0, len(audio) - n + 1, n):
        yield audio[i : i + n]
