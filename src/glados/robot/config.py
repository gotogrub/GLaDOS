from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, HttpUrl


class VisionSettings(BaseModel):
    camera_index: int = 0
    capture_interval_seconds: float = 3.0
    scene_change_threshold: float = 0.05
    max_tokens: int = 64
    resolution: int = 384
    save_frames: bool = False
    save_frames_dir: str = "vision_frames"
    save_frames_max: int = 1000


class FaceDisplaySettings(BaseModel):
    enabled: bool = False
    assets_dir: str = "assets/faces"
    default_emotion: str = "neutral"
    monitor: int = 0
    width: int = 0
    height: int = 0


class AutonomySettings(BaseModel):
    cooldown_s: float = 5.0
    tick_prompt: str = "Сцена: {scene}\nЛица: {faces}"


class FaceProfile(BaseModel):
    """Profile for a known face."""
    name: str
    description: str = ""


class RobotConfig(BaseModel):
    llm_model: str
    completion_url: HttpUrl
    api_key: str | None = None
    llm_options: dict[str, Any] | None = None
    asr_engine: str = "whisper"
    voice: str = "glados_ru"
    personality: str = "Ты — ГЛаДОС из Portal. Саркастичный ИИ. Отвечай кратко, на русском."
    knowledge_path: str | None = "data/knowledge.json"
    face_db: str = "faces/"
    face_names: dict[str, str | FaceProfile] | None = None
    vision: VisionSettings = VisionSettings()
    autonomy: AutonomySettings = AutonomySettings()
    face_display: FaceDisplaySettings = FaceDisplaySettings()
    interruptible: bool = True
    interrupt_keywords: list[str] | None = None
    tools_enabled: bool = False
    announcement: str | None = None

    def get_face_profiles(self) -> dict[str, FaceProfile]:
        """Normalize face_names to FaceProfile dict."""
        if not self.face_names:
            return {}
        result: dict[str, FaceProfile] = {}
        for folder, val in self.face_names.items():
            if isinstance(val, str):
                result[folder] = FaceProfile(name=val)
            else:
                result[folder] = val
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        robot = data["Robot"]

        # Collect env overrides: first check real env vars, then .env file.
        # Real env vars take priority over .env, .env takes priority over yaml.
        env_overrides: dict[str, str] = {}

        # Load .env file into os.environ (setdefault — won't override real env vars)
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip()
                if value:
                    key = key.strip()
                    os.environ.setdefault(key, value)
                    env_overrides[key] = os.environ[key]

        # Real env vars override .env values
        for key in ("OLLAMA_URL", "OLLAMA_API_KEY", "OLLAMA_MODEL"):
            val = os.environ.get(key, "")
            if val:
                env_overrides[key] = val

        # Apply overrides to yaml config
        _ENV_TO_YAML = {
            "OLLAMA_URL": "completion_url",
            "OLLAMA_API_KEY": "api_key",
            "OLLAMA_MODEL": "llm_model",
        }
        for env_key, yaml_key in _ENV_TO_YAML.items():
            if env_key in env_overrides:
                robot[yaml_key] = env_overrides[env_key]

        return cls.model_validate(robot)
