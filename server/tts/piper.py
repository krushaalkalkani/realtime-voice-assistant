"""Local TTS with Piper (fast neural voices that run on CPU)."""

import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator

from piper import PiperVoice

from server.tts.base import TTS

_DONE = object()


class PiperTTS(TTS):
    name = "piper"

    def __init__(self, voice_path: Path):
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path}. Download it with:\n"
                f"  uv run python -m piper.download_voices {voice_path.stem} --data-dir {voice_path.parent}"
            )
        self.voice = PiperVoice.load(voice_path)
        self.sample_rate = self.voice.config.sample_rate

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        # Piper's synthesize() is a blocking generator that yields one audio chunk
        # per sentence. We run it in a background thread and hand each chunk to the
        # async world through a queue, so audio can be sent while later sentences
        # are still being synthesized.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for chunk in self.voice.synthesize(text):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.audio_int16_bytes)
            except Exception as e:  # pass errors to the async side
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
