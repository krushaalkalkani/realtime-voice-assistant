# Latency results

Benchmark: the same 8 spoken answers replayed into every config (cold start each run). Target: **< 1500 ms** from the user stopping talking to the first audio.

## Stage breakdown — final config (`9-final`, n=12 turns)

| Stage | p50 (ms) | p95 (ms) |
|---|---|---|
| Endpointing (VAD silence) | 672 | 672 |
| ASR final | 499 | 621 |
| LLM TTFT | 123 | 181 |
| LLM text wait | 296 | 380 |
| TTS first byte | 162 | 234 |
| Overhead | 39 | 46 |
| Pipeline (speech end → audio) | 1129 | 1352 |
| **Total to first audio (user-perceived)** | 1801 | 2024 |

Target 1500 ms at p50: **missed ❌** (p95 2024 ms).

## Optimizations (rows 1-8 are cumulative: each adds one change to the row above; `9-final` is the shipped combination, measured on its own, Δ vs baseline)

| Config | Change | Perceived p50 | p95 | Δ p50 vs previous | First turn | ASR WER |
|---|---|---|---|---|---|---|
| `1-baseline` | Phase 1 pipeline | 3378 | 5654 | — | 6928 | 3.3% |
| `2-sentence-tts` | + sentence-level TTS streaming | 1845 | 2861 | -1534 ms | 3851 | 3.3% |
| `3-warmup` | + warm model loading | 1972 | 3024 | +127 ms | 1734 | 3.3% |
| `4-short-prompt` | + shorter system prompt | 2196 | 3691 | +225 ms | 3480 | 3.3% |
| `5-tiny-asr` | + smaller ASR model (tiny.en) | 1854 | 3745 | -342 ms | 2943 | 6.0% |
| `6-no-partials` | + no partial ASR while speaking | 1860 | 3500 | +6 ms | 2961 | 6.0% |
| `7-vad-500` | + shorter end-of-speech silence (500 ms) | 1882 | 4666 | +22 ms | 4568 | 6.0% |
| `8-fast-first` | + first audio chunk ends at a comma | 1771 | 3110 | -111 ms | 1688 | 6.0% |
| `9-final` | Chosen defaults: sentence TTS + warm-up + fast first chunk | 1801 | 2024 | -1577 ms | 1871 | 3.3% |

## Per-stage p50 for every config (ms)

| Config | Endpointing (VAD silence) | ASR final | LLM TTFT | LLM text wait | TTS first byte | Overhead | Perceived |
|---|---|---|---|---|---|---|---|
| `1-baseline` | 672 | 468 | 119 | 1798 | 127 | 40 | 3378 |
| `2-sentence-tts` | 672 | 471 | 114 | 346 | 143 | 38 | 1845 |
| `3-warmup` | 672 | 490 | 130 | 368 | 158 | 39 | 1972 |
| `4-short-prompt` | 672 | 462 | 122 | 587 | 185 | 43 | 2196 |
| `5-tiny-asr` | 672 | 241 | 122 | 562 | 198 | 41 | 1854 |
| `6-no-partials` | 672 | 300 | 138 | 606 | 162 | 42 | 1860 |
| `7-vad-500` | 480 | 304 | 135 | 613 | 201 | 44 | 1882 |
| `8-fast-first` | 480 | 481 | 189 | 256 | 155 | 40 | 1771 |
| `9-final` | 672 | 499 | 123 | 296 | 162 | 39 | 1801 |
