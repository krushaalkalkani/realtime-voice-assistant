"""Small building blocks for resilience: time limits and circuit breakers."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Awaitable, TypeVar

T = TypeVar("T")


class StageFailure(Exception):
    """A pipeline stage failed. reason: "timeout" | "error" | "circuit_open"."""

    def __init__(self, stage: str, reason: str, detail: str = "", elapsed_ms: float = 0.0,
                 partial: str = ""):
        super().__init__(f"{stage} {reason}: {detail}")
        self.stage, self.reason, self.detail = stage, reason, detail
        self.elapsed_ms = round(elapsed_ms, 1)
        self.partial = partial  # LLM text produced before the failure, if any


async def with_timeout(stage: str, aw: Awaitable[T], timeout_s: float) -> T:
    """Await `aw`, converting timeouts and errors into StageFailure."""
    t0 = time.perf_counter()
    try:
        return await asyncio.wait_for(aw, timeout_s)
    except asyncio.TimeoutError:
        raise StageFailure(stage, "timeout", f"no result after {timeout_s:g}s",
                           (time.perf_counter() - t0) * 1000) from None
    except StageFailure:
        raise
    except Exception as e:
        raise StageFailure(stage, "error", f"{type(e).__name__}: {e}",
                           (time.perf_counter() - t0) * 1000) from e


async def stream_with_timeouts(stage: str, it: AsyncIterator[T], first_s: float,
                               next_s: float) -> AsyncIterator[T]:
    """Re-yield items from `it`, but give up if the first item takes longer than `first_s`
    or any later gap is longer than `next_s`."""
    it = aiter(it)
    first = True
    try:
        while True:
            try:
                item = await with_timeout(stage, anext(it), first_s if first else next_s)
            except StageFailure as f:
                if isinstance(f.__cause__, StopAsyncIteration):
                    return
                raise
            first = False
            yield item
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose:
            try:
                await aclose()
            except BaseException:
                pass


class CircuitBreaker:
    """After a failure, skip the component for `cooldown_s` instead of waiting for it to
    time out again on every turn. After the cooldown one trial call is allowed
    ("half-open"); success closes the breaker, failure opens it again."""

    def __init__(self, name: str, cooldown_s: float, clock=time.monotonic):
        self.name, self.cooldown_s, self.clock = name, cooldown_s, clock  # replay swaps the clock
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None and self.clock() - self.opened_at < self.cooldown_s

    def record_failure(self) -> None:
        self.opened_at = self.clock()

    def record_success(self) -> None:
        self.opened_at = None
