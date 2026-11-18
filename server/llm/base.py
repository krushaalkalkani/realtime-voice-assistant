"""LLM interface. Any chat model (Ollama, Anthropic, ...) implements this."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLM(ABC):
    name: str = "base"

    @abstractmethod
    def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Yields text pieces (tokens) as soon as the model produces them."""

    async def is_warm(self) -> bool | None:
        """Is the model loaded and ready? None = can't tell (e.g. a cloud API)."""
        return None
