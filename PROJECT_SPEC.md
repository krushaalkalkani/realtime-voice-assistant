# Project 5 — Real-Time Voice Assistant with Latency Budget and Resilience

> **For Claude (the builder):** Build phase by phase. After each phase, stop, show how to run it, and explain what was built in simple language — the owner (Krushal) is learning through this project. Keep every external service behind an interface so it can be swapped (cloud ↔ local). Default to free/local components where possible so the project costs ~$0 to run; paid services are optional upgrades.

## 1. What we are building
A **streaming voice assistant**: the user speaks → speech-to-text (ASR) → LLM reasoning → text-to-speech (TTS) → audio plays back, all streamed in real time over WebSockets. It comes with a **per-request latency breakdown**, graceful degradation when components fail, and a **replay mode** for debugging.

(The video offers 3 tracks — voice assistant, webcam vision pipeline, streaming log analyzer. Voice is chosen because voice AI is growing fast and tooling is mature.)

**Default persona: "Mock interview coach"** — it asks interview questions out loud, listens to the answer, and gives short spoken feedback. Useful for the owner's own job search, and a memorable demo.

## 2. Why it matters
Shows you can handle streaming data, tight latency budgets and messy real-time systems — not just batch "happy path" code. Being able to say *"total response is 1.2s, and here's the exact breakdown"* is what excites interviewers.

## 3. Tech stack
| Layer | Default (free/local) | Upgrade (paid, optional) |
|---|---|---|
| Transport | FastAPI WebSockets | — |
| Client | Browser page (Web Audio API, mic capture, audio playback) | — |
| VAD / turn detection | Silero VAD | — |
| ASR | `faster-whisper` (small/base) or `whisper.cpp` | Deepgram streaming |
| LLM | Ollama local model (streaming) | Anthropic API (`claude-haiku-5-5`) |
| TTS | Piper or Kokoro (local) | ElevenLabs / Cartesia |
| Metrics | per-stage timestamps → SQLite/JSONL | — |
| Visualization | Streamlit page or in-browser waterfall chart | — |

## 4. Architecture
```
 Browser mic ──audio frames──► WS server
                                 ├─ VAD: detect end of user speech
                                 ├─ ASR (streaming/partial) ── transcript
                                 ├─ LLM (streaming tokens) ── sentence chunker
                                 ├─ TTS per sentence (streaming) ── audio chunks
 Browser speaker ◄──audio chunks─┘
                                 └─ event bus: every stage emits timestamped events
                                        → latency budget per turn → dashboard
 Recorder: saves input audio + events per session → replay mode
```
Key idea: **pipeline overlap** — start TTS on the first complete sentence while the LLM is still generating.

## 5. Suggested repo structure
```
realtime-voice-assistant/
├── README.md
├── server/
│   ├── main.py            # FastAPI + WebSocket
│   ├── events.py          # typed events (pydantic) with timestamps
│   ├── vad.py
│   ├── asr/{base.py, faster_whisper.py, deepgram.py}
│   ├── llm/{base.py, ollama.py, anthropic.py}
│   ├── tts/{base.py, piper.py, elevenlabs.py}
│   ├── pipeline.py        # orchestration, timeouts, fallbacks
│   └── metrics.py
├── web/{index.html, app.js}
├── replay/{recordings/, replay.py}
├── dashboard/latency.py
└── tests/
```

## 6. Phases

### Phase 1 — End-to-end streaming pipeline
- Browser captures mic audio → WebSocket → VAD → ASR → LLM (streaming) → TTS → audio back to browser.
- Structured, typed events for every step (`speech_start`, `speech_end`, `asr_partial`, `asr_final`, `llm_first_token`, `llm_done`, `tts_first_byte`, `audio_playback_start`, …).
- Don't optimize yet — just get it working end to end.

Acceptance: hold a multi-turn spoken conversation in the browser.

### Phase 2 — Latency budget
- Decompose every turn into: **ASR latency**, **LLM time-to-first-token**, **TTS time-to-first-byte**, network/overhead, and **total "user stops talking → assistant starts talking"**.
- Per-request waterfall visualization + p50/p95 across sessions.
- Optimize with evidence: sentence-level TTS streaming, smaller ASR model, warm model loading, shorter system prompt — measure each change.
- Set a target budget (e.g. < 1.5s to first audio) and report whether you hit it.

Acceptance: README shows a latency breakdown chart and a before/after optimization table.

### Phase 3 — Resilience
- **Timeouts** on every stage; nothing blocks forever.
- **Graceful degradation:** ASR down → tell the user and offer text input; LLM timeout → short spoken "give me a second" filler, then retry or fall back to a smaller/local model; TTS down → stream text to the screen.
- **Barge-in:** if the user starts speaking, stop playback.
- **Replay mode:** record input audio + events; re-run any session through the pipeline deterministically for debugging.
- **Fault-injection script** that randomly kills/delays components; document observed behavior.

Acceptance: with a component forcibly failed, the assistant degrades instead of hanging, and the failure is visible in the metrics.

## 7. Results tables (fill with real numbers)
| Stage | p50 (ms) | p95 (ms) |
|---|---|---|
| ASR final | | |
| LLM TTFT | | |
| TTS first byte | | |
| Overhead | | |
| **Total to first audio** | | |

| Optimization | Total p50 before | after |
|---|---|---|
| Sentence-level TTS streaming | | |
| Smaller ASR model | | |

| Fault injected | Behavior | Recovered? |
|---|---|---|
| ASR down | | |
| LLM timeout | | |
| TTS down | | |

## 8. Definition of done
- [ ] Real-time spoken conversation in the browser
- [ ] Per-turn latency breakdown + p50/p95 dashboard
- [ ] Measured optimizations
- [ ] Timeouts, fallbacks, barge-in
- [ ] Replay mode + fault-injection results
- [ ] README with demo video/GIF, latency chart, failure-handling table

## 9. Resume bullet template
- Built a real-time voice assistant (WebSockets, Whisper ASR, streaming LLM, TTS) with per-stage latency tracing; cut median time-to-first-audio from **X s to Y s** via sentence-level streaming and model tuning.
- Designed graceful degradation, barge-in, and a replay/fault-injection harness so component failures degrade service instead of hanging.

## 10. Interview questions this prepares you for
- Where does the latency go in a voice pipeline? How did you reduce it?
- Why WebSockets instead of HTTP requests?
- How do you handle a slow or failed dependency in real time?
- How do you debug a real-time system after the fact?
