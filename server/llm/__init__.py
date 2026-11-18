from server.config import Settings
from server.llm.base import LLM


def build_llm(s: Settings) -> LLM:
    if s.llm_provider == "ollama":
        from server.llm.ollama import OllamaLLM

        return OllamaLLM(s.llm_model, s.ollama_url, s.llm_max_tokens, s.llm_seed)
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
