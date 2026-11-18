"""All tunable settings in one place.

Every value can be overridden with an environment variable of the same name,
e.g. `LLM_MODEL=phi4-mini uv run uvicorn server.main:app --port 8010`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):  # bool("0") would be True, so parse explicitly
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return type(default)(raw)


SYSTEM_PROMPTS = {
    # Phase 1 prompt (~100 tokens, invites 2-4 sentence replies)
    "long": (
        "You are a friendly but honest mock interview coach for software and AI "
        "engineering jobs. You are SPEAKING out loud, so: reply in 2 to 4 short "
        "sentences, plain conversational English, no lists, no markdown, no emojis. "
        "After the candidate answers, give one specific piece of praise or one "
        "concrete improvement, then ask the next interview question. Ask one "
        "question at a time. Mix behavioral and technical questions."
    ),
    # Phase 2 prompt (~45 tokens, asks for 2 short sentences)
    "short": (
        "You are a mock interview coach for software and AI jobs, speaking out loud. "
        "Reply in exactly 2 short plain sentences: one specific piece of feedback, "
        "then the next interview question."
    ),
}


@dataclass(frozen=True)
class Settings:
    # --- audio coming from the browser ---
    sample_rate: int = 16_000          # browser resamples mic audio to 16 kHz mono
    frame_samples: int = 512           # Silero VAD works on 512-sample (32 ms) frames

    # --- VAD / turn detection ---
    vad_threshold: float = _env("VAD_THRESHOLD", 0.5)        # prob above this = speech
    vad_neg_threshold: float = _env("VAD_NEG_THRESHOLD", 0.35)  # prob below this = silence
    vad_start_frames: int = _env("VAD_START_FRAMES", 3)      # ~96 ms of speech to trigger
    vad_min_silence_ms: int = _env("VAD_MIN_SILENCE_MS", 700)  # silence that ends a turn
    vad_min_speech_ms: int = _env("VAD_MIN_SPEECH_MS", 250)    # shorter blips are ignored
    vad_preroll_ms: int = _env("VAD_PREROLL_MS", 300)          # audio kept from before the trigger
    vad_max_speech_s: int = _env("VAD_MAX_SPEECH_S", 30)       # hard cap on one utterance

    # --- ASR ---
    asr_provider: str = _env("ASR_PROVIDER", "faster_whisper")
    asr_model: str = _env("ASR_MODEL", "base.en")
    asr_partial_every_ms: int = _env("ASR_PARTIAL_EVERY_MS", 1000)  # 0 disables partials

    # --- LLM ---
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    llm_model: str = _env("LLM_MODEL", "llama3.2:3b")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    llm_max_tokens: int = _env("LLM_MAX_TOKENS", 200)
    prompt_style: str = _env("PROMPT_STYLE", "long")          # "long" | "short"
    llm_seed: int = _env("LLM_SEED", -1)                       # fixed seed = reproducible replies (benchmarks)

    # --- TTS ---
    tts_provider: str = _env("TTS_PROVIDER", "piper")
    piper_voice: str = _env("PIPER_VOICE", "en_US-lessac-medium")
    # "full": wait for the whole LLM reply, then speak it.
    # "sentence": speak each sentence as soon as the LLM finishes it (pipeline overlap).
    tts_mode: str = _env("TTS_MODE", "sentence")  # Phase 2 benchmark: -1.7 s p50 vs "full"
    sentence_min_chars: int = _env("SENTENCE_MIN_CHARS", 10)  # don't speak tiny fragments alone
    # In sentence mode, let the FIRST chunk end at a comma/colon/dash too, so audio starts sooner.
    tts_fast_first: bool = _env("TTS_FAST_FIRST", True)
    fast_first_min_chars: int = _env("FAST_FIRST_MIN_CHARS", 20)

    # --- latency ---
    warmup: bool = _env("WARMUP", True)           # run each model once at startup (first turn -2 s)
    latency_target_ms: int = _env("LATENCY_TARGET_MS", 1500)  # user stops talking -> audio starts
    run_label: str = _env("RUN_LABEL", "live")    # tags sessions so benchmarks can be compared

    # --- resilience (Phase 3) ---
    # Every stage has a time limit, so nothing can block a turn forever.
    asr_timeout_s: float = _env("ASR_TIMEOUT_S", 3.0)             # base limit ...
    asr_timeout_per_audio_s: float = _env("ASR_TIMEOUT_PER_AUDIO_S", 0.5)  # ... + 0.5 s per s of speech
    llm_first_token_timeout_s: float = _env("LLM_FIRST_TOKEN_TIMEOUT_S", 3.0)
    llm_stall_timeout_s: float = _env("LLM_STALL_TIMEOUT_S", 4.0)        # max gap between tokens
    llm_cold_timeout_s: float = _env("LLM_COLD_TIMEOUT_S", 10.0)  # first-token limit when the model must load
    llm_fallback_model: str = _env("LLM_FALLBACK_MODEL", "phi4-mini")    # "" disables the fallback
    llm_fallback_timeout_s: float = _env("LLM_FALLBACK_TIMEOUT_S", 8.0)  # may need to load the model
    tts_timeout_s: float = _env("TTS_TIMEOUT_S", 3.0)                    # per audio chunk
    playback_grace_s: float = _env("PLAYBACK_GRACE_S", 5.0)  # resume listening if the browser never confirms
    breaker_cooldown_s: float = _env("BREAKER_COOLDOWN_S", 30.0)  # skip a failed component this long
    barge_in: bool = _env("BARGE_IN", True)                    # user can talk over the coach
    barge_in_min_speech_ms: int = _env("BARGE_IN_MIN_SPEECH_MS", 400)  # speech needed to interrupt
    record_audio: bool = _env("RECORD_AUDIO", True)            # save mic audio for replay
    faults: str = _env("FAULTS", "")                           # e.g. "llm=hang,tts=down" (testing)

    # canned lines used when components fail
    filler_text: str = "Give me a second."
    asr_failed_text: str = (
        "Sorry, I couldn't make that out. My speech recognition is having trouble, "
        "so you can type your answer in the box below."
    )
    llm_failed_text: str = "Sorry, I'm having trouble thinking right now. Could you say that again?"

    # --- paths ---
    models_dir: Path = ROOT / "models"
    sessions_dir: Path = Path(_env("SESSIONS_DIR", str(ROOT / "logs" / "sessions")))
    web_dir: Path = ROOT / "web"

    # --- persona ---
    greeting: str = (
        "Hi, I'm your mock interview coach. I'll ask you a question, listen to your "
        "answer, and give you quick feedback. Let's start. Tell me a bit about yourself."
    )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.prompt_style]

    def snapshot(self) -> dict:
        """The settings that affect latency - stored with every session for comparison."""
        return {
            "label": self.run_label, "asr_model": self.asr_model, "llm_model": self.llm_model,
            "tts_mode": self.tts_mode, "tts_fast_first": self.tts_fast_first, "warmup": self.warmup, "prompt_style": self.prompt_style,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "asr_partial_every_ms": self.asr_partial_every_ms,
            "llm_fallback_model": self.llm_fallback_model, "barge_in": self.barge_in,
            "faults": self.faults,
            # resilience settings (replay re-applies these)
            "barge_in_min_speech_ms": self.barge_in_min_speech_ms, "breaker_cooldown_s": self.breaker_cooldown_s,
            "asr_timeout_s": self.asr_timeout_s, "asr_timeout_per_audio_s": self.asr_timeout_per_audio_s,
            "llm_first_token_timeout_s": self.llm_first_token_timeout_s,
            "llm_stall_timeout_s": self.llm_stall_timeout_s, "llm_fallback_timeout_s": self.llm_fallback_timeout_s,
            "llm_cold_timeout_s": self.llm_cold_timeout_s, "tts_timeout_s": self.tts_timeout_s,
            "playback_grace_s": self.playback_grace_s, "sentence_min_chars": self.sentence_min_chars,
            "fast_first_min_chars": self.fast_first_min_chars,
            "vad_threshold": self.vad_threshold, "vad_neg_threshold": self.vad_neg_threshold,
            "vad_start_frames": self.vad_start_frames, "vad_min_speech_ms": self.vad_min_speech_ms,
            "vad_preroll_ms": self.vad_preroll_ms,
        }


settings = Settings()
