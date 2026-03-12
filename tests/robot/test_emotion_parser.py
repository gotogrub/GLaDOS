from glados.robot.text_pipeline import EmotionParser


def test_emotion_detected_sarcasm():
    p = EmotionParser()
    out = p.feed("[SARCASM] ")
    out += p.feed("Hello world")
    assert p.emotion == "sarcasm"
    assert "Hello world" in out
    assert "[SARCASM]" not in out


def test_emotion_detected_anger():
    p = EmotionParser()
    out = p.feed("[ANG")
    assert out == ""  # still buffering
    out += p.feed("ER] Shut up")
    assert p.emotion == "anger"
    assert "Shut up" in out


def test_emotion_default_neutral():
    p = EmotionParser()
    assert p.emotion == "neutral"


def test_emotion_no_tag_passes_through():
    p = EmotionParser()
    # Text without a tag — parser gives up after _MAX_BUFFER chars
    result = ""
    for word in "Ну привет, как дела у тебя сегодня, дорогой?".split():
        result += p.feed(word + " ")
    assert p.emotion == "neutral"
    assert "привет" in result


def test_emotion_unknown_tag_passes_through():
    p = EmotionParser()
    out = p.feed("[UNKNOWN] text")
    # UNKNOWN not in EMOTIONS — parser gives up at ']'
    assert p.emotion == "neutral"
    assert "[UNKNOWN]" in out


def test_emotion_reset():
    p = EmotionParser()
    p.feed("[SARCASM] hi")
    assert p.emotion == "sarcasm"
    p.reset()
    assert p.emotion == "neutral"
    assert not p._detected


def test_emotion_tag_with_leading_whitespace():
    p = EmotionParser()
    out = p.feed("  [CURIOSITY] What is this?")
    assert p.emotion == "curiosity"
    assert "What is this?" in out


def test_emotion_all_valid_tags():
    for tag in EmotionParser.EMOTIONS:
        p = EmotionParser()
        p.feed(f"[{tag}] text")
        assert p.emotion == tag.lower(), f"Failed for {tag}"


def test_emotion_passthrough_after_detection():
    p = EmotionParser()
    p.feed("[AMUSEMENT] first")
    out = p.feed(" second chunk")
    assert out == " second chunk"
    assert p.emotion == "amusement"
