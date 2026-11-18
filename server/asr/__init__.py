from server.asr.base import ASR
from server.config import Settings


def build_asr(s: Settings) -> ASR:
    if s.asr_provider == "faster_whisper":
        from server.asr.faster_whisper import FasterWhisperASR

        return FasterWhisperASR(s.asr_model, str(s.models_dir / "whisper"))
    raise ValueError(f"Unknown ASR_PROVIDER: {s.asr_provider}")
