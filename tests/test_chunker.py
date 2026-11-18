from server.chunker import SentenceChunker


def test_splits_on_sentence_end_while_streaming():
    c = SentenceChunker(min_chars=10)
    out = []
    for tok in ["Great ", "answer, ", "really. ", "Now, ", "tell me ", "about a ", "bug you fixed?"]:
        out += c.push(tok)
    assert out == ["Great answer, really."]
    assert c.flush() == ["Now, tell me about a bug you fixed?"]


def test_short_sentences_are_merged():
    c = SentenceChunker(min_chars=10)
    assert c.push("Okay. That was solid. ") == ["Okay. That was solid."]


def test_decimals_do_not_split():
    c = SentenceChunker(min_chars=5)
    assert c.push("Latency went from 1.5 seconds to 0.9 seconds. Nice") == [
        "Latency went from 1.5 seconds to 0.9 seconds."
    ]


def test_fast_first_chunk_breaks_at_comma_only_once():
    c = SentenceChunker(min_chars=10, first_clause_min_chars=20)
    out = c.push("That was a clear answer, and I liked the example, a lot. Next question, please? ")
    assert out == ["That was a clear answer,", "and I liked the example, a lot.", "Next question, please?"]


def test_fast_first_waits_for_enough_text():
    c = SentenceChunker(min_chars=10, first_clause_min_chars=20)
    assert c.push("Good, ") == []                     # too short to speak alone
    assert c.push("you covered the basics, ") == ["Good, you covered the basics,"]
