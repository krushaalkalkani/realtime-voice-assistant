"""Local ASR with faster-whisper (Whisper re-implemented on CTranslate2; runs on CPU)."""

import asyncio

import numpy as np
from faster_whisper import WhisperModel

from server.asr.base import ASR


class FasterWhisperASR(ASR):
    name = "faster_whisper"

    def __init__(self, model: str, download_root: str):
        # int8 = quantized weights: much faster on CPU, tiny accuracy cost.
        # num_workers=2 lets a partial and a final transcription run at the same time.
        self.model = WhisperModel(
            model, device="cpu", compute_type="int8",
            download_root=download_root, num_workers=2,
        )

    def _transcribe_blocking(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,                       # greedy decoding
            vad_filter=False,                  # we already did VAD ourselves
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        # Whisper is blocking CPU work -> run it in a thread so the event loop stays free.
        return await asyncio.to_thread(self._transcribe_blocking, audio)
