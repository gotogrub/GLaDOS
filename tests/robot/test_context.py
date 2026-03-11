from glados.robot.config import FaceProfile
from glados.robot.context import ContextBuilder


def test_context_injects_vision():
    profiles = {"creator": FaceProfile(name="Создатель", description="Твой создатель")}
    cb = ContextBuilder(face_profiles=profiles)
    messages = [{"role": "system", "content": "personality"}]
    result = cb.build(
        messages=messages,
        vision_desc="A person. Лица: creator",
        autonomy=False,
    )
    vision_msgs = [m for m in result if "Сейчас ты видишь:" in m.get("content", "")]
    assert len(vision_msgs) == 1
    assert "Создатель" in vision_msgs[0]["content"]
    assert "Твой создатель" in vision_msgs[0]["content"]


def test_context_no_vision_when_empty():
    cb = ContextBuilder()
    messages = [{"role": "user", "content": "hi"}]
    result = cb.build(messages=messages, vision_desc=None, autonomy=False)
    assert not any("Сейчас ты видишь:" in m.get("content", "") for m in result)


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
    profiles = {"creator": FaceProfile(name="Создатель", description="Твой создатель")}
    cb = ContextBuilder(face_profiles=profiles)
    messages = [{"role": "system", "content": "personality"}]
    # No "creator" in vision desc → no face name injection
    result = cb.build(
        messages=messages,
        vision_desc="A room with a table",
        autonomy=False,
    )
    vision_msgs = [m for m in result if "Сейчас ты видишь:" in m.get("content", "")]
    assert len(vision_msgs) == 1
    assert "Создатель" not in vision_msgs[0]["content"]


def test_context_face_profile_without_description():
    profiles = {"bob": FaceProfile(name="Боб")}
    cb = ContextBuilder(face_profiles=profiles)
    result = cb.build(
        messages=[{"role": "system", "content": "p"}],
        vision_desc="Лица: bob",
        autonomy=False,
    )
    vision_msgs = [m for m in result if "Сейчас ты видишь:" in m.get("content", "")]
    assert "Боб" in vision_msgs[0]["content"]
    # No " — " separator when description is empty
    assert "Боб —" not in vision_msgs[0]["content"]
