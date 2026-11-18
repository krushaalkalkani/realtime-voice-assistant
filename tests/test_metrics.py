from server.metrics import percentiles, turn_breakdown


def ev(type, t, **data):
    return {"type": type, "t_ms": t, "turn_id": 1, "data": data}


def test_breakdown_adds_up():
    events = [
        ev("speech_end", 1000, trailing_silence_ms=700),
        ev("asr_final", 1300),
        ev("llm_start", 1305),
        ev("llm_first_token", 1450),
        ev("tts_start", 1700),
        ev("tts_first_byte", 1800),
        ev("audio_playback_start", 1850),
    ]
    b = turn_breakdown(events)
    assert b["asr_ms"] == 300
    assert b["llm_ttft_ms"] == 145
    assert b["llm_wait_ms"] == 250
    assert b["tts_ttfb_ms"] == 100
    assert b["total_ms"] == 850
    assert b["overhead_ms"] == 55  # 5 ms gap + 50 ms delivery
    assert b["perceived_ms"] == 1550


def test_incomplete_turn_has_no_breakdown():
    assert turn_breakdown([ev("speech_end", 0), ev("asr_final", 10)]) is None


def test_percentiles():
    rows = [{"total_ms": v} for v in range(1, 101)]
    p = percentiles(rows, ["total_ms"])["total_ms"]
    assert round(p["p50"]) == 50 and round(p["p95"]) == 95 and p["n"] == 100
