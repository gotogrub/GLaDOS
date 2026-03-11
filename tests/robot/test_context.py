from glados.robot.context import ContextBuilder


def test_context_injects_vision():
    cb = ContextBuilder(face_names={"creator": "твой создатель"})
    messages = [{"role": "system", "content": "personality"}]
    result = cb.build(
        messages=messages,
        vision_desc="A person. Лица: creator",
        autonomy=False,
    )
    vision_msgs = [m for m in result if "[vision]" in m.get("content", "")]
    assert len(vision_msgs) == 1
    assert "твой создатель" in vision_msgs[0]["content"]


def test_context_no_vision_when_empty():
    cb = ContextBuilder()
    messages = [{"role": "user", "content": "hi"}]
    result = cb.build(messages=messages, vision_desc=None, autonomy=False)
    assert not any("[vision]" in m.get("content", "") for m in result)


def test_context_autonomy_prompt():
    cb = ContextBuilder(autonomy_prompt="Act autonomously.")
    messages = [{"role": "system", "content": "personality"}]
    result = cb.build(messages=messages, vision_desc=None, autonomy=True)
    assert any("Act autonomously" in m.get("content", "") for m in result)


def test_context_no_autonomy_prompt_when_not_autonomy():
    cb = ContextBuilder(autonomy_prompt="Act autonomously.")
    messages = [{"role": "system", "content": "personality"}]
    result = cb.build(messages=messages, vision_desc=None, autonomy=False)
    assert not any("Act autonomously" in m.get("content", "") for m in result)


def test_context_face_names_only_when_present():
    cb = ContextBuilder(face_names={"creator": "твой создатель"})
    messages = [{"role": "system", "content": "personality"}]
    # No "creator" in vision desc → no face name injection
    result = cb.build(
        messages=messages,
        vision_desc="A room with a table",
        autonomy=False,
    )
    vision_msgs = [m for m in result if "[vision]" in m.get("content", "")]
    assert len(vision_msgs) == 1
    assert "создатель" not in vision_msgs[0]["content"]
