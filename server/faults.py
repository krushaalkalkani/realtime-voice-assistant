"""Fault injection: break components on purpose to test that the pipeline degrades gracefully.

Each real component is wrapped (FaultyASR / FaultyLLM / FaultyTTS). The wrappers do nothing
until a fault is switched on - via the FAULTS env var, POST /faults, or the page's fault panel.

Modes (spec string per component):
    ok              normal
    down            fail immediately (like a crashed service / connection refused)
    slow:<ms>       add a delay before answering (e.g. slow:5000)
    hang            never answer (the caller's timeout must save us)
    stall           LLM/TTS only: start answering, then freeze mid-way
    flaky:<p>       fail with probability p (e.g. flaky:0.5)

Example:  FAULTS="llm=hang,tts=down" uv run uvicorn server.main:app --port 8010
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from server.asr.base import ASR
from server.llm.base import LLM
from server.tts.base import TTS

COMPONENTS = ("asr", "llm", "tts")


class InjectedFault(ConnectionError):
    """Raised by a component that was switched to 'down' / 'flaky'."""


@dataclass
class Fault:
    mode: str = "ok"
    value: float = 0.0  # delay in ms for "slow", probability for "flaky"

    @classmethod
    def parse(cls, spec: str) -> "Fault":
        mode, _, value = spec.strip().partition(":")
        if mode not in ("ok", "down", "slow", "hang", "stall", "flaky"):
            raise ValueError(f"unknown fault mode: {spec!r}")
        default = {"slow": 3000.0, "flaky": 0.5}.get(mode, 0.0)
        return cls(mode, float(value) if value else default)

    def __str__(self) -> str:
        return self.mode + (f":{self.value:g}" if self.mode in ("slow", "flaky") else "")


class FaultInjector:
    def __init__(self, spec: str = ""):
        self.faults = {c: Fault() for c in COMPONENTS}
        self.update(spec)

    def update(self, spec: str | dict) -> None:
        """spec: "asr=down,llm=slow:5000" or {"asr": "down", "llm": "slow:5000"}."""
        if isinstance(spec, str):
            spec = dict(p.split("=", 1) for p in spec.split(",") if "=" in p)
        for comp, s in spec.items():
            comp = comp.strip()
            if comp not in self.faults:
                raise ValueError(f"unknown component: {comp!r}")
            self.faults[comp] = Fault.parse(s)

    def describe(self) -> dict[str, str]:
        return {c: str(f) for c, f in self.faults.items()}

    async def before_call(self, comp: str) -> None:
        """Applied before a component starts working."""
        f = self.faults[comp]
        if f.mode == "down":
            raise InjectedFault(f"injected fault: {comp} down")
        if f.mode == "flaky" and random.random() < f.value:
            raise InjectedFault(f"injected fault: {comp} flaky")
        if f.mode == "slow":
            await asyncio.sleep(f.value / 1000)
        if f.mode == "hang":
            await asyncio.Event().wait()  # forever

    def stalls(self, comp: str) -> bool:
        return self.faults[comp].mode == "stall"


async def _stall_forever():
    await asyncio.Event().wait()


class FaultyASR(ASR):
    def __init__(self, inner: ASR, inj: FaultInjector):
        self.inner, self.inj, self.name = inner, inj, inner.name

    async def transcribe(self, audio: np.ndarray) -> str:
        await self.inj.before_call("asr")
        if self.inj.stalls("asr"):
            await _stall_forever()
        return await self.inner.transcribe(audio)


class FaultyLLM(LLM):
    def __init__(self, inner: LLM, inj: FaultInjector):
        self.inner, self.inj, self.name = inner, inj, inner.name
        self.model = getattr(inner, "model", "")

    async def is_warm(self):
        return await self.inner.is_warm()

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        await self.inj.before_call("llm")
        n = 0
        async for piece in self.inner.stream(messages):
            yield piece
            n += 1
            if n == 6 and self.inj.stalls("llm"):  # a few tokens, then freeze
                await _stall_forever()


class FaultyTTS(TTS):
    def __init__(self, inner: TTS, inj: FaultInjector):
        self.inner, self.inj, self.name = inner, inj, inner.name
        self.sample_rate = inner.sample_rate

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        await self.inj.before_call("tts")
        n = 0
        async for chunk in self.inner.stream(text):
            yield chunk
            n += 1
            if self.inj.stalls("tts"):  # first chunk of each text, then freeze
                await _stall_forever()
