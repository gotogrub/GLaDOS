import os
from pathlib import Path
import tempfile
import yaml


def test_robot_config_from_yaml(monkeypatch):
    # Clear env vars so .env doesn't override test values
    for key in ("OLLAMA_URL", "OLLAMA_API_KEY", "OLLAMA_MODEL"):
        monkeypatch.delenv(key, raising=False)
    # Prevent .env file from being read
    monkeypatch.chdir(tempfile.gettempdir())
    data = {
        "Robot": {
            "llm_model": "gemma3:4b",
            "completion_url": "http://localhost:11434/api/chat",
            "llm_options": {"num_ctx": 4096},
            "asr_engine": "whisper",
            "voice": "glados_ru",
            "personality": "Test personality.",
            "knowledge_path": "data/knowledge.json",
            "face_db": "faces/",
            "vision": {
                "camera_index": 0,
                "capture_interval_seconds": 3,
                "scene_change_threshold": 0.05,
                "max_tokens": 64,
            },
            "autonomy": {
                "cooldown_s": 5,
                "tick_prompt": "Scene: {scene}\nFaces: {faces}",
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    from glados.robot.config import RobotConfig
    config = RobotConfig.from_yaml(path)
    assert config.llm_model == "gemma3:4b"
    assert config.vision.camera_index == 0
    assert config.autonomy.cooldown_s == 5
    assert config.face_db == "faces/"
    Path(path).unlink()


def test_robot_config_defaults():
    from glados.robot.config import RobotConfig
    config = RobotConfig(
        llm_model="gemma3:4b",
        completion_url="http://localhost:11434/api/chat",
        asr_engine="whisper",
        voice="glados_ru",
        personality="Test.",
    )
    assert config.face_db == "faces/"
    assert config.vision.capture_interval_seconds == 3
    assert config.autonomy.cooldown_s == 5
