"""TTS interface. Any text-to-speech engine (Piper, ElevenLabs, ...) implements this."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTS(ABC):
    name: str = "base"
    sample_rate: int  # sample rate of the PCM audio this engine produces

    @abstractmethod
    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yields raw audio: 16-bit signed little-endian PCM, mono, at self.sample_rate."""
