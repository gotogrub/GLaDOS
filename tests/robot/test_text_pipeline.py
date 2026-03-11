from glados.robot.text_pipeline import ThinkFilter, SentenceSplitter


def test_think_filter_strips_tags():
    f = ThinkFilter()
    assert f.feed("<think>reasoning</think>Hello") == "Hello"


def test_think_filter_multipart():
    f = ThinkFilter()
    assert f.feed("<think>start") == ""
    assert f.feed(" middle") == ""
    assert f.feed("</think>World") == "World"


def test_think_filter_passthrough():
    f = ThinkFilter()
    assert f.feed("no tags here") == "no tags here"


def test_think_filter_thinking_tag():
    f = ThinkFilter()
    assert f.feed("<thinking>hmm</thinking>Ok") == "Ok"


def test_sentence_splitter_basic():
    s = SentenceSplitter()
    results = []
    for token in ["Привет", ".", " Как", " дела", "?"]:
        sentence = s.feed(token)
        if sentence:
            results.append(sentence)
    remaining = s.flush()
    if remaining:
        results.append(remaining)
    assert results == ["Привет.", "Как дела?"]


def test_sentence_splitter_exclamation():
    s = SentenceSplitter()
    r = s.feed("Wow!")
    assert r == "Wow!"


def test_sentence_splitter_flush_partial():
    s = SentenceSplitter()
    assert s.feed("Hello") is None
    assert s.feed(" world") is None
    assert s.flush() == "Hello world"


def test_sentence_splitter_empty_flush():
    s = SentenceSplitter()
    assert s.flush() == ""
