from glados.robot.text_pipeline import EmotionParser


def test_new_format_with_intensity():
    p = EmotionParser()
    out = p.feed("[emotion:sarcastic,0.8] Hello")
    assert p.emotion == "sarcastic"
    assert p.intensity == 0.8
    assert "Hello" in out
    assert "[emotion:" not in out


def test_new_format_without_intensity():
    p = EmotionParser()
    out = p.feed("[emotion:angry] Shut up")
    assert p.emotion == "angry"
    assert p.intensity == 0.5  # default
    assert "Shut up" in out


def test_new_format_split_tokens():
    p = EmotionParser()
    out = p.feed("[emotion:cond")
    assert out == ""
    out += p.feed("escending,0.7] Text")
    assert p.emotion == "condescending"
    assert p.intensity == 0.7
    assert "Text" in out


def test_legacy_format_sarcasm():
    p = EmotionParser()
    out = p.feed("[SARCASM] Hello")
    assert p.emotion == "sarcastic"
    assert "Hello" in out


def test_legacy_format_anger():
    p = EmotionParser()
    out = p.feed("[ANGER] Go away")
    assert p.emotion == "angry"


def test_legacy_format_all_mapped():
    for old, new in EmotionParser._OLD_EMOTION_MAP.items():
        p = EmotionParser()
        p.feed(f"[{old}] text")
        assert p.emotion == new, f"Failed: {old} → expected {new}, got {p.emotion}"


def test_default_emotion():
    p = EmotionParser()
    assert p.emotion == "sarcastic"
    assert p.intensity == 0.5


def test_no_tag_passes_through():
    p = EmotionParser()
    result = ""
    for word in "Привет как дела у тебя сегодня дорогой друг мой?".split():
        result += p.feed(word + " ")
    assert p.emotion == "sarcastic"  # default
    assert "Привет" in result


def test_unknown_tag_passes_through():
    p = EmotionParser()
    out = p.feed("[UNKNOWN] text")
    assert p.emotion == "sarcastic"  # default
    assert "[UNKNOWN]" in out


def test_reset():
    p = EmotionParser()
    p.feed("[emotion:angry,0.9] hi")
    assert p.emotion == "angry"
    assert p.intensity == 0.9
    p.reset()
    assert p.emotion == "sarcastic"
    assert p.intensity == 0.5
    assert not p._detected


def test_all_new_emotions():
    for emotion in EmotionParser.EMOTIONS:
        p = EmotionParser()
        p.feed(f"[emotion:{emotion},0.5] text")
        assert p.emotion == emotion, f"Failed for {emotion}"


def test_passthrough_after_detection():
    p = EmotionParser()
    p.feed("[emotion:amused,0.6] first")
    out = p.feed(" second chunk")
    assert out == " second chunk"


def test_flush_unreleased_buffer():
    p = EmotionParser()
    p.feed("[emotion:bor")  # incomplete, still buffering
    leftover = p.flush()
    assert "[emotion:bor" in leftover
    assert p._detected
