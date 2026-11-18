from server.config import Settings
from server.tts.base import TTS


def build_tts(s: Settings) -> TTS:
    if s.tts_provider == "piper":
        from server.tts.piper import PiperTTS

        return PiperTTS(s.models_dir / "piper" / f"{s.piper_voice}.onnx")
    raise ValueError(f"Unknown TTS_PROVIDER: {s.tts_provider}")
