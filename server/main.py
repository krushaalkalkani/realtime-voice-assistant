"""FastAPI app: serves the web page and the /ws WebSocket.

Run:  uv run uvicorn server.main:app --port 8010
Open: http://localhost:8010
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.asr import build_asr
from server.config import settings
from server.faults import FaultInjector, FaultyASR, FaultyLLM, FaultyTTS
from server.llm import build_llm
from server.pipeline import Session
from server.tts import build_tts

log = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

components: dict = {}
faults = FaultInjector(settings.faults)  # all "ok" unless FAULTS=... or POST /faults


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE at startup (not per connection) - loading takes seconds.
    log.info("Loading ASR (%s %s)...", settings.asr_provider, settings.asr_model)
    # Each component is wrapped by the fault injector (a no-op unless a fault is switched on).
    components["asr"] = FaultyASR(build_asr(settings), faults)
    log.info("Loading TTS (%s %s)...", settings.tts_provider, settings.piper_voice)
    components["tts"] = FaultyTTS(build_tts(settings), faults)
    components["llm"] = FaultyLLM(build_llm(settings), faults)
    # The fallback LLM is NOT wrapped: it's the independent plan B when the primary is broken.
    components["llm_fallback"] = (
        build_llm(dataclasses.replace(settings, llm_model=settings.llm_fallback_model))
        if settings.llm_fallback_model else None
    )
    if settings.faults:
        log.warning("FAULT INJECTION ACTIVE: %s", faults.describe())
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            tags = (await c.get(f"{settings.ollama_url}/api/tags")).json()
        names = [m["name"] for m in tags.get("models", [])]
        if settings.llm_model not in names:
            log.warning("Ollama model %s not found. Run: ollama pull %s", settings.llm_model, settings.llm_model)
    except Exception:
        log.warning("Ollama not reachable at %s - start it with `ollama serve`", settings.ollama_url)
    if settings.warmup:
        await warmup()
    log.info("Ready -> open http://localhost:8010")
    yield


async def warmup():
    """Run every model once so the first real turn doesn't pay for cold start
    (Ollama loading weights into GPU memory, first-call init in Whisper / Piper)."""
    t0 = time.perf_counter()
    await components["asr"].transcribe(np.zeros(16_000, dtype=np.float32))
    async for _ in components["tts"].stream("Warming up."):
        pass
    try:
        # Same system prompt as real turns, so Ollama also caches its processed prefix.
        msgs = [{"role": "system", "content": settings.system_prompt}, {"role": "user", "content": "Hi"}]
        # Only the PRIMARY model: on a 16 GB Mac, Ollama keeps one of these models in memory at a
        # time, so warming the fallback would evict the primary (found by fault injection).
        async for _ in components["llm"].stream(msgs):
            break  # one token is enough; leaving the loop cancels the rest
    except Exception as e:
        log.warning("LLM warm-up failed: %s", e)
    log.info("Warm-up done in %.0f ms", (time.perf_counter() - t0) * 1000)


app = FastAPI(title="Realtime Voice Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(settings.web_dir / "index.html")


@app.get("/faults")
async def get_faults():
    return faults.describe()


@app.post("/faults")
async def set_faults(spec: dict[str, str]):
    """Switch faults on/off at runtime, e.g. {"llm": "hang", "tts": "ok"}. For testing."""
    try:
        faults.update(spec)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.warning("Faults now: %s", faults.describe())
    return faults.describe()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()  # several tasks send on this socket; keep messages whole

    async def send_json(msg: dict):
        async with send_lock:
            await ws.send_text(json.dumps(msg))

    async def send_bytes(data: bytes):
        async with send_lock:
            await ws.send_bytes(data)

    session = Session(
        settings, components["asr"], components["llm"], components["tts"], send_json, send_bytes,
        llm_fallback=components["llm_fallback"], faults_state=faults.describe,
    )
    log.info("Session %s connected (%s)", session.id, settings.snapshot())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.handle_audio(msg["bytes"])      # mic audio
            elif msg.get("text") is not None:
                await session.handle_control(json.loads(msg["text"]))  # control messages
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        log.info("Session %s closed (events: %s)", session.id, session.bus.log_path)
