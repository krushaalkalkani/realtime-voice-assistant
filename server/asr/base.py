"""ASR interface. Any speech-to-text engine (local Whisper, Deepgram, ...) implements this."""

from abc import ABC, abstractmethod

import numpy as np


class ASR(ABC):
    name: str = "base"

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> str:
        """audio: float32 mono samples at 16 kHz in [-1, 1]. Returns the transcript text."""
