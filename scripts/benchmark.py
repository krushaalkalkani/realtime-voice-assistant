"""Latency benchmark: runs the SAME spoken conversation under several configs.

For each config it:
  1. unloads the Ollama model (so every run starts cold, like a fresh deploy)
  2. starts the server with that config's env vars
  3. plays N recorded answers into it through the fake browser
  4. stops the server
Sessions land in logs/bench/<label>/; then run `scripts/report.py` for tables + charts.

Configs are CUMULATIVE: each one keeps the previous optimizations and adds one more,
so the difference between two neighbours is the effect of one change.

Usage:
    uv run python scripts/benchmark.py                 # all configs, 12 turns each
    uv run python scripts/benchmark.py --turns 4 --only 1-baseline 2-sentence-tts
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time

import httpx

from fake_browser import ROOT, FakeBrowser, speak

PORT = 8011
BENCH_DIR = ROOT / "logs" / "bench"

# The scripted "candidate" answers (also the ground truth for ASR word error rate).
ANSWERS = [
    "I'm a software engineer focused on AI applications. Recently I built a real time voice assistant in Python.",
    "In my last project the biggest challenge was latency. I measured every stage and fixed the slowest one first.",
    "I handle disagreements by listening first. Then I write down the trade offs so we can decide with data.",
    "A hash map gives constant time lookups on average, but it uses more memory than a sorted array.",
    "I would start with a simple baseline, add monitoring, and only then optimize the parts that matter.",
    "My greatest weakness is that I sometimes over polish things, so now I set a time limit for each task.",
    "Retrieval augmented generation means we search our documents first, then give the relevant text to the model.",
    "I want this role because I enjoy building products that people actually use every day.",
]

BASE = {
    "TTS_MODE": "full", "WARMUP": "0", "PROMPT_STYLE": "long", "ASR_MODEL": "base.en",
    "ASR_PARTIAL_EVERY_MS": "1000", "VAD_MIN_SILENCE_MS": "700", "TTS_FAST_FIRST": "0",
    "LLM_SEED": "42",  # same sampling seed everywhere -> differences come from the config, not luck
}
CONFIGS: list[tuple[str, str, dict]] = [
    ("1-baseline", "Phase 1 pipeline", {}),
    ("2-sentence-tts", "+ sentence-level TTS streaming", {"TTS_MODE": "sentence"}),
    ("3-warmup", "+ warm model loading", {"WARMUP": "1"}),
    ("4-short-prompt", "+ shorter system prompt", {"PROMPT_STYLE": "short"}),
    ("5-tiny-asr", "+ smaller ASR model (tiny.en)", {"ASR_MODEL": "tiny.en"}),
    ("6-no-partials", "+ no partial ASR while speaking", {"ASR_PARTIAL_EVERY_MS": "0"}),
    ("7-vad-500", "+ shorter end-of-speech silence (500 ms)", {"VAD_MIN_SILENCE_MS": "500"}),
    ("8-fast-first", "+ first audio chunk ends at a comma", {"TTS_FAST_FIRST": "1"}),
]


# The configuration we actually ship, chosen from the cumulative results above
# (keeps what helped, drops what hurt or cost accuracy). Measured on its own.
FINAL = ("9-final", "Chosen defaults: sentence TTS + warm-up + fast first chunk", {
    "TTS_MODE": "sentence", "WARMUP": "1", "TTS_FAST_FIRST": "1",
})
ALL_RUNS = CONFIGS + [FINAL]


def cumulative_env(label: str) -> dict:
    if label == FINAL[0]:
        return {**BASE, **FINAL[2]}
    env = dict(BASE)
    for name, _, changes in CONFIGS:
        env.update(changes)
        if name == label:
            return env
    raise KeyError(label)


def unload_ollama(model="llama3.2:3b"):
    try:
        httpx.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        time.sleep(1)
    except Exception as e:
        print("  (could not unload ollama model:", e, ")")


def start_server(label: str, env_changes: dict) -> subprocess.Popen:
    out_dir = BENCH_DIR / label
    shutil.rmtree(out_dir, ignore_errors=True)
    env = {**os.environ, **env_changes, "RUN_LABEL": label, "SESSIONS_DIR": str(out_dir)}
    log = open(ROOT / "logs" / f"bench_server_{label}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(300):  # model downloads on first use can take a while
        time.sleep(0.5)
        try:
            if httpx.get(f"http://localhost:{PORT}/", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"server for {label} crashed; see logs/bench_server_{label}.log")
    raise TimeoutError("server did not start")


async def run_conversation(n_turns: int, voices) -> int:
    ok = 0
    async with FakeBrowser(f"ws://localhost:{PORT}/ws", verbose=False) as b:
        if not await b.wait_turn(0):
            return 0
        for i in range(1, n_turns + 1):
            if await b.turn(i, voices[(i - 1) % len(voices)]):
                ok += 1
            print(f"    turn {i}: heard {b.transcripts.get(i, '')!r}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--only", nargs="*", help="config labels to run")
    args = ap.parse_args()

    print("Synthesizing the candidate's answers...")
    voices = [speak(a) for a in ANSWERS]

    for label, desc, _ in ALL_RUNS:
        if args.only and label not in args.only:
            continue
        env = cumulative_env(label)
        print(f"\n=== {label}: {desc}\n    {env}")
        unload_ollama()
        proc = start_server(label, env)
        try:
            ok = asyncio.run(run_conversation(args.turns, voices))
            print(f"  completed {ok}/{args.turns} turns")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    print("\nDone. Now run: uv run python scripts/report.py")


if __name__ == "__main__":
    main()
