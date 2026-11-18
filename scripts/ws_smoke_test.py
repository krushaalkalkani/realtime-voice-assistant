"""Headless end-to-end test against the running server (two spoken turns).

Usage (server must be running):
    uv run python scripts/ws_smoke_test.py
"""

import asyncio
import wave

from fake_browser import ROOT, FakeBrowser, speak

ANSWERS = [
    "I'm a software engineer. I build AI applications with Python and I recently made a voice assistant.",
    "I would measure the latency of every stage, then optimize the slowest one first.",
]


async def main(url="ws://localhost:8010/ws"):
    async with FakeBrowser(url) as b:
        print("Turn 0 (greeting)")
        assert await b.wait_turn(0)
        for i, answer in enumerate(ANSWERS, start=1):
            print(f"\nTurn {i}: speaking -> {answer!r}")
            if not await b.turn(i, speak(answer)):
                print("FAILED")
                return
            print(f"           heard: {b.transcripts.get(i)!r}")
            print(f"           coach: {b.replies.get(i, '').strip()!r}")

    out = ROOT / "logs"
    for t, pcm in b.audio.items():
        with wave.open(str(out / f"smoke_reply_{t}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(b.sample_rate)
            w.writeframes(bytes(pcm))
    print(f"\nOK - multi-turn conversation worked. Reply audio saved to {out}/smoke_reply_*.wav")


if __name__ == "__main__":
    asyncio.run(main())
