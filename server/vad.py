"""Voice Activity Detection (VAD) + turn detection.

Two layers:
  * SileroVAD   - a tiny neural net. Give it 32 ms of audio, it returns the
                  probability (0..1) that someone is speaking.
  * TurnDetector - turns that stream of probabilities into two decisions:
                  "the user STARTED talking" and "the user FINISHED talking".
                  "Finished" = enough continuous silence (vad_min_silence_ms).

We reuse the Silero ONNX model that ships inside faster-whisper, so no extra
download (and no PyTorch) is needed.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from server.config import Settings

CONTEXT_SAMPLES = 64  # Silero expects the last 64 samples of the previous frame prepended


def _default_model_path() -> Path:
    import faster_whisper

    return Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"


class SileroVAD:
    """Streaming wrapper: keeps the model's hidden state between frames."""

    def __init__(self, model_path: Path | None = None, frame_samples: int = 512):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(
            str(model_path or _default_model_path()),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.frame_samples = frame_samples
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """frame: float32 array of exactly `frame_samples` samples at 16 kHz."""
        assert frame.shape == (self.frame_samples,), frame.shape
        x = np.concatenate([self._context, frame])[None, :].astype(np.float32)
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(prob.reshape(-1)[0])


class TurnDetector:
    """Feeds frames to the VAD and decides when an utterance starts and ends.

    process(frame) returns one of:
        None       - nothing changed
        "start"    - user started speaking
        "end"      - user finished speaking; call take_utterance() to get the audio
        "discard"  - it was just a short blip (cough, click); ignore it
    """

    def __init__(self, vad: SileroVAD, s: Settings):
        self.vad = vad
        self.s = s
        frame_ms = s.frame_samples / s.sample_rate * 1000
        self._frame_ms = frame_ms
        self._silence_frames_needed = int(s.vad_min_silence_ms / frame_ms)
        self._min_speech_frames = int(s.vad_min_speech_ms / frame_ms)
        self._max_frames = int(s.vad_max_speech_s * 1000 / frame_ms)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(s.vad_preroll_ms / frame_ms)))
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self.speaking = False
        self._above = 0          # consecutive frames above threshold (before start)
        self._silence = 0        # consecutive silent frames (while speaking)
        self._speech_frames = 0  # frames that actually contained speech
        self._buffer: list[np.ndarray] = []
        self.last_prob = 0.0
        self.trailing_silence_ms = 0.0

    def process(self, frame: np.ndarray) -> str | None:
        prob = self.vad(frame)
        self.last_prob = prob

        if not self.speaking:
            self._preroll.append(frame)
            self._above = self._above + 1 if prob >= self.s.vad_threshold else 0
            if self._above >= self.s.vad_start_frames:
                self.speaking = True
                self._buffer = list(self._preroll)  # include audio from just before the trigger
                self._speech_frames = self._above
                self._silence = 0
                return "start"
            return None

        # currently speaking
        self._buffer.append(frame)
        if prob >= self.s.vad_threshold:
            self._silence = 0
            self._speech_frames += 1
        elif prob < self.s.vad_neg_threshold:
            self._silence += 1
        # probabilities in between: neither reset nor count silence (hysteresis)

        if self._silence >= self._silence_frames_needed or len(self._buffer) >= self._max_frames:
            # how long the user had already been quiet when we decided (endpointing delay)
            self.trailing_silence_ms = self._silence * self._frame_ms
            self.speaking = False
            self._above = 0
            if self._speech_frames < self._min_speech_frames:
                self._buffer = []
                return "discard"
            return "end"
        return None

    def current_audio(self) -> np.ndarray:
        """Audio captured so far in the ongoing utterance (used for partial ASR)."""
        return np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.float32)

    def take_utterance(self) -> np.ndarray:
        audio = self.current_audio()
        self._buffer = []
        self._preroll.clear()
        return audio

    @property
    def speech_frames_ms(self) -> float:
        """How much actual speech (not silence) the current utterance contains so far."""
        return self._speech_frames * self._frame_ms if self.speaking else 0.0

    @property
    def speech_ms(self) -> float:
        return len(self._buffer) * self._frame_ms
