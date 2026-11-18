"""Local LLM served by Ollama (http://localhost:11434), streamed token by token."""

import json
from typing import AsyncIterator

import httpx

from server.llm.base import LLM


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str, base_url: str, max_tokens: int, seed: int = -1):
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(60.0, connect=5.0))

    async def is_warm(self) -> bool | None:
        """Ask Ollama which models are in memory. A cold model needs seconds to load."""
        try:
            resp = await self.client.get("/api/ps", timeout=0.5)
            return any(m.get("name") == self.model for m in resp.json().get("models", []))
        except Exception:
            return None

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"num_predict": self.max_tokens, "temperature": 0.7},
        }
        if self.seed >= 0:
            payload["options"]["seed"] = self.seed
        # Ollama streams one JSON object per line: {"message": {"content": "..."}, "done": false}
        async with self.client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
