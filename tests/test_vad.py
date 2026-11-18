import numpy as np

from server.config import settings
from server.vad import SileroVAD, TurnDetector
from tests.conftest import frames


def run(audio):
    td = TurnDetector(SileroVAD(), settings)
    return [r for f in frames(audio) if (r := td.process(f))], td


def test_silence_never_triggers():
    results, _ = run(np.zeros(16_000 * 3, dtype=np.float32))
    assert results == []


def test_noise_never_triggers():
    rng = np.random.default_rng(0)
    results, _ = run((rng.standard_normal(16_000 * 3) * 0.01).astype(np.float32))
    assert results == []


def test_speech_then_silence_gives_start_then_end(speech_16k):
    silence = np.zeros(16_000, dtype=np.float32)
    audio = np.concatenate([silence, speech_16k, silence, silence])
    results, _ = run(audio)
    assert results == ["start", "end"]


def test_utterance_contains_the_speech(speech_16k):
    silence = np.zeros(16_000, dtype=np.float32)
    td = TurnDetector(SileroVAD(), settings)
    for f in frames(np.concatenate([silence, speech_16k, silence, silence])):
        if td.process(f) == "end":
            utt = td.take_utterance()
            # roughly the speech length (+ preroll + trailing silence), not the whole input
            assert 0.8 * len(speech_16k) < len(utt) < len(speech_16k) + 1.5 * 16_000
            return
    raise AssertionError("no end detected")
