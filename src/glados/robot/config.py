from __future__ import annotations

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


class AutonomySettings(BaseModel):
    cooldown_s: float = 5.0
    tick_prompt: str = "Сцена: {scene}\nЛица: {faces}"


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
    face_names: dict[str, str] | None = None
    vision: VisionSettings = VisionSettings()
    autonomy: AutonomySettings = AutonomySettings()
    interruptible: bool = True
    interrupt_keywords: list[str] | None = None
    tools_enabled: bool = False
    announcement: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data["Robot"])
