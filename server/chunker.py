"""Sentence chunker: cuts the LLM token stream into speakable pieces.

    push("Great ") -> []
    push("answer. Next") -> ["Great answer."]
    flush() -> ["Next"]

This is what lets TTS start on sentence 1 while the LLM is still writing sentence 2.

Optional "fast first chunk": the very first piece may also end at a clause break
(comma, colon, semicolon, dash) once it is long enough. The first chunk is the one
the user is waiting for; every later chunk is synthesized while earlier audio plays.
"""

import re

# sentence end: . ! or ? (possibly repeated / followed by a closing quote) and then whitespace
_SENTENCE = re.compile(r"[.!?]+[\"')\]]*\s+")
_CLAUSE = re.compile(r"(?:[.!?]+[\"')\]]*|[,;:]|\s[-–—])\s+")


class SentenceChunker:
    def __init__(self, min_chars: int = 10, first_clause_min_chars: int | None = None):
        self.min_chars = min_chars  # avoid speaking "Okay." or "Mr." as a separate chunk
        self.first_clause_min_chars = first_clause_min_chars
        self._buf = ""
        self._emitted = 0

    def push(self, text: str) -> list[str]:
        self._buf += text
        out = []
        start = 0
        while True:
            first = self._emitted == 0 and self.first_clause_min_chars is not None
            pattern, min_chars = (_CLAUSE, self.first_clause_min_chars) if first else (_SENTENCE, self.min_chars)
            for m in pattern.finditer(self._buf, start):
                candidate = self._buf[start:m.end()].strip()
                if len(candidate) >= min_chars:
                    out.append(candidate)
                    start = m.end()
                    self._emitted += 1
                    break
            else:
                break
        self._buf = self._buf[start:]
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        if rest:
            self._emitted += 1
        return [rest] if rest else []
