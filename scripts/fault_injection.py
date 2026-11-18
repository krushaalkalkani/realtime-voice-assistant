"""Fault injection against the REAL pipeline (Whisper + Ollama + Piper).

For each scenario: open a fresh session, switch a fault on, speak an answer, and record what
the assistant did. Then switch the fault off, wait for the circuit breaker cooldown, speak
again, and check the assistant recovered. A final "random" scenario flips random faults on
every turn. A last scenario points the server at a dead Ollama port (a real outage, nothing
injected).

Output: docs/fault_results.md  (and the sessions in logs/faults/, replayable)

Usage:  uv run python scripts/fault_injection.py [--random-turns 8]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import subprocess
import sys
import time

import httpx

from fake_browser import ROOT, FakeBrowser, speak

PORT = 8012
OUT_DIR = ROOT / "logs" / "faults"
COOLDOWN_S = 3.0
ANSWER = "I'm a software engineer. Recently I built a real time voice assistant in Python."
RECOVERY = "I would measure the latency of every stage, then fix the slowest one first."
INTERRUPT = "Sorry, can I stop you there and ask a question?"

SCENARIOS = [
    ("ASR down", {"asr": "down"}),
    ("ASR hangs", {"asr": "hang"}),
    ("LLM down", {"llm": "down"}),
    ("LLM hangs (timeout)", {"llm": "hang"}),
    ("LLM slow (5 s)", {"llm": "slow:5000"}),
    ("LLM stalls mid-reply", {"llm": "stall"}),
    ("TTS down", {"tts": "down"}),
    ("TTS hangs", {"tts": "hang"}),
    ("LLM + TTS down", {"llm": "down", "tts": "down"}),
]
ALL_OK = {"asr": "ok", "llm": "ok", "tts": "ok"}


def start_server(extra_env: dict) -> subprocess.Popen:
    env = {**os.environ, "SESSIONS_DIR": str(OUT_DIR), "RUN_LABEL": "faults", "LLM_SEED": "42",
           "BREAKER_COOLDOWN_S": str(COOLDOWN_S), **extra_env}
    log = open(ROOT / "logs" / "fault_server.log", "a")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(PORT)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(0.5)
        try:
            if httpx.get(f"http://localhost:{PORT}/faults", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise RuntimeError("server crashed; see logs/fault_server.log")
    raise TimeoutError("server did not start")


def set_faults(spec: dict):
    httpx.post(f"http://localhost:{PORT}/faults", json={**ALL_OK, **spec}, timeout=5).raise_for_status()


def describe(b: FakeBrowser, turn: int) -> dict:
    """What the user experienced in one turn, reconstructed from the events."""
    ev = b.turn_events(turn)
    first = {}
    for e in ev:
        first.setdefault(e["type"], e)
    start = first.get("speech_end") or first.get("text_input")
    # first thing the user got: sound, or (text-only turns) the reply starting to appear on screen
    text_start = [e for e in ev if e["type"] == "llm_first_token"]
    response = first.get("audio_playback_start") or (text_start[-1] if text_start else first.get("tts_done"))
    failures = [f"{e['data']['stage']} {e['data']['reason']}" + (f" after {e['data']['elapsed_ms']:.0f} ms"
                if e["data"].get("elapsed_ms") else "") for e in ev if e["type"] == "stage_failed"]
    actions = [e["data"]["action"] for e in ev if e["type"] == "fallback"]
    return {
        "status": b.done.get(turn, "hung"),
        "failures": failures, "actions": actions,
        "heard_audio": "audio_playback_start" in first,
        "response_ms": round(response["t_ms"] - start["t_ms"]) if start and response else None,
        "reply": b.replies.get(turn, "").strip(),
    }


async def one_scenario(name: str, spec: dict, voices: dict) -> dict:
    async with FakeBrowser(f"ws://localhost:{PORT}/ws", verbose=False) as b:
        await b.wait_turn(0, timeout=30)
        set_faults(spec)
        ok = True
        try:
            await b.turn(1, voices["answer"], timeout=45)
        except TimeoutError:
            ok = False
        faulty = describe(b, 1)
        faulty["finished"] = ok
        set_faults({})
        await asyncio.sleep(COOLDOWN_S + 0.5)  # let the circuit breaker close again
        try:
            await b.turn(2, voices["recovery"], timeout=45)
        except TimeoutError:
            pass
        rec = describe(b, 2)
    # Recovered = the next turn works with the primary components again. Saying "give me a
    # second" while the primary LLM reloads (evicted by the fallback model) still counts.
    recovered = rec["status"] == "ok" and not rec["failures"] and rec["heard_audio"]
    if recovered and "filler_while_model_loads" in rec["actions"]:
        recovered = "reload"
    return {"name": name, "spec": spec, "faulty": faulty, "recovery": rec, "recovered": recovered}


async def barge_in_scenario(voices: dict) -> dict:
    async with FakeBrowser(f"ws://localhost:{PORT}/ws", verbose=False, realtime_playback=True) as b:
        await b.wait_turn(0, timeout=30)  # greeting plays in real time
        await b.say(voices["answer"])
        await b.wait_event("audio_playback_start", 1, timeout=30)
        await asyncio.sleep(1.0)  # let the coach talk for a second, then interrupt
        await b.say(voices["interrupt"], lead_s=0.0)
        await b.wait_turn(2, timeout=60)
        interrupted = b.done.get(1) == "interrupted"
        barge = next((e for e in b.events if e["type"] == "barge_in"), None)
        stop = next((m for m in b.messages if m["kind"] == "stop_audio"), None)
        return {"name": "Barge-in (user talks over the coach)", "spec": {},
                "faulty": {"status": b.done.get(1), "failures": [], "actions": ["stop_audio_and_listen"] if stop else [],
                           "heard_audio": True, "response_ms": None, "finished": interrupted,
                           "reply": b.replies.get(1, "").strip(), "barge_ms": barge["data"]["speech_ms"] if barge else None},
                "recovery": describe(b, 2), "recovered": interrupted and b.done.get(2) == "ok"}


async def random_scenario(voices: dict, turns: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    modes = {"asr": ["ok", "ok", "down", "hang", "slow:2000"],
             "llm": ["ok", "ok", "down", "hang", "slow:5000", "stall"],
             "tts": ["ok", "ok", "down", "hang", "stall"]}
    rows = []
    async with FakeBrowser(f"ws://localhost:{PORT}/ws", verbose=False) as b:
        await b.wait_turn(0, timeout=30)
        for t in range(1, turns + 1):
            spec = {c: rng.choice(m) for c, m in modes.items()}
            set_faults(spec)
            t0 = time.perf_counter()
            try:
                await b.turn(t, voices["answer"] if t % 2 else voices["recovery"], timeout=45)
                hung = False
            except TimeoutError:
                hung = True
            rows.append({"turn": t, "spec": {k: v for k, v in spec.items() if v != "ok"},
                         "hung": hung, "seconds": time.perf_counter() - t0, **describe(b, t)})
    set_faults({})
    return rows


def fmt_behavior(d: dict) -> str:
    parts = []
    if d["failures"]:
        parts.append("; ".join(d["failures"]))
    if d["actions"]:
        parts.append("→ " + ", ".join(a.replace("_", " ") for a in d["actions"]))
    if d.get("response_ms") is not None:
        parts.append(f"user got a response after {d['response_ms'] / 1000:.1f} s"
                     + ("" if d["heard_audio"] else " (text appears on screen)"))
    if d["reply"]:
        parts.append(f'said: "{d["reply"][:70]}{"…" if len(d["reply"]) > 70 else ""}"')
    return "<br>".join(parts) or d["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-turns", type=int, default=8)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Synthesizing the candidate's voice...")
    voices = {"answer": speak(ANSWER), "recovery": speak(RECOVERY), "interrupt": speak(INTERRUPT)}

    results, rand = [], []
    proc = start_server({})
    try:
        for name, spec in SCENARIOS:
            print(f"- {name} ...", flush=True)
            r = asyncio.run(one_scenario(name, spec, voices))
            print(f"    {fmt_behavior(r['faulty']).replace('<br>', ' | ')}  recovered={r['recovered']}")
            results.append(r)
        print("- Barge-in ...", flush=True)
        r = asyncio.run(barge_in_scenario(voices))
        print(f"    interrupted={r['faulty']['finished']} recovered={r['recovered']}")
        results.append(r)
        print(f"- Random faults for {args.random_turns} turns ...", flush=True)
        rand = asyncio.run(random_scenario(voices, args.random_turns))
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print("- Real outage: Ollama unreachable ...", flush=True)
    proc = start_server({"OLLAMA_URL": "http://localhost:9"})  # nothing listens there
    try:
        r = asyncio.run(one_scenario("Ollama unreachable (real outage, not injected)", {}, voices))
        # no 'recovery' possible here: Ollama stays unreachable for this server
        r["recovered"] = None
        results.append(r)
        print(f"    {fmt_behavior(r['faulty']).replace('<br>', ' | ')}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    md = ["# Fault injection results\n",
          "Real Whisper + Ollama + Piper, fault switched on for one spoken turn, then off; "
          f"circuit-breaker cooldown {COOLDOWN_S:g} s for the test. Generated by `scripts/fault_injection.py`.\n",
          "| Fault injected | Behavior | Recovered? |", "|---|---|---|"]
    for r in results:
        rec = {True: "✅ next turn normal", "reload": "✅ next turn normal, after a filler while the main model reloads",
               False: "❌", None: "n/a (outage persists)"}[r["recovered"]]
        if r["faulty"].get("barge_ms") is not None:
            beh = (f"coach stopped after {r['faulty']['barge_ms']} ms of user speech; "
                   f"partial reply kept in history; the interruption became the next turn")
        else:
            beh = fmt_behavior(r["faulty"])
        if not r["faulty"].get("finished", True):
            beh = "**HUNG** — " + beh
        md.append(f"| {r['name']} | {beh} | {rec} |")

    if rand:
        md += ["\n## Random faults, one session\n", "| Turn | Faults | Took | Outcome |", "|---|---|---|---|"]
        for row in rand:
            faults = ", ".join(f"{k}={v}" for k, v in row["spec"].items()) or "none"
            outcome = "**HUNG**" if row["hung"] else (", ".join(a.replace("_", " ") for a in row["actions"]) or "normal")
            md.append(f"| {row['turn']} | {faults} | {row['seconds']:.1f} s | {outcome} |")
        hung = sum(r["hung"] for r in rand)
        md.append(f"\n{len(rand) - hung}/{len(rand)} turns finished; longest {max(r['seconds'] for r in rand):.1f} s "
                  "(includes the user's ~5 s of speech).")
    (ROOT / "docs" / "fault_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
