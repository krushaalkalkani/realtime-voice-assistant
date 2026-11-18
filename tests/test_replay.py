"""Replay reproduces a recorded session, and shows the effect of a changed setting."""

import asyncio
import dataclasses
import sys

import numpy as np

from server.config import settings
from server.pipeline import Session
from tests.conftest import ROOT, frames
from tests.test_resilience import FAST, FakeASR, FakeLLM, FakeTTS

sys.path.insert(0, str(ROOT / "replay"))
import replay as rp  # noqa: E402


def record_session(tmp_path, speech):
    """A live-like session: two spoken turns, browser confirms playback after each."""
    s = dataclasses.replace(settings, sessions_dir=tmp_path, **FAST)

    async def go():
        async def sj(m): pass
        async def sb(b): pass
        sess = Session(s, FakeASR(), FakeLLM(), FakeTTS(), sj, sb)
        await sess.start()
        await sess._turn_task
        await sess.handle_control({"type": "playback_end", "turn_id": 0})
        silence = np.zeros(16000, np.float32)
        for _ in range(2):
            for f in frames(np.concatenate([silence, speech, silence, silence])):
                await sess.handle_audio((f * 32767).astype("<i2").tobytes())
                if sess._turn_task and not sess._turn_task.done():
                    await sess._turn_task
                    await sess.handle_control({"type": "playback_end", "turn_id": sess.responding_turn})
        await sess.close()
        return sess.id

    return asyncio.run(go())


def test_replay_matches_recording_and_reflects_changes(speech_16k, tmp_path, monkeypatch):
    sid = record_session(tmp_path, speech_16k)
    monkeypatch.setattr(rp, "SEARCH_DIRS", [tmp_path])
    monkeypatch.setattr(rp, "REPLAY_DIR", tmp_path / "replays")

    rec, new, _ = asyncio.run(rp.replay(sid, {}, "", "recorded", "silent", "recorded"))
    assert len(rec) == 2 and [rp.comparable(r) for r in new] == [rp.comparable(r) for r in rec]

    # same recording, but with the speech recognizer broken -> both turns degrade
    _, broken, _ = asyncio.run(rp.replay(sid, {}, "asr=down", "recorded", "silent", "recorded"))
    assert [r["flags"] for r in broken] == [["asr:ask_to_type"]] * 2
