from __future__ import annotations

from typing import Final

# Instructions for the LLM to handle vision messages from the vision module.
# These instructions are essential for proper integration of vision observations into the conversation.
SYSTEM_PROMPT_VISION_HANDLING: Final[str] = (
    "У тебя есть камера. Сообщения с префиксом [vision] — это описание того, что ты видишь прямо сейчас. "
    "Используй эту информацию в своих ответах. Можешь комментировать увиденное."
)

# Default prompts for FastVLM inference.
VISION_DEFAULT_PROMPT: Final[str] = "Describe the image briefly, focusing on salient elements."
VISION_DETAIL_PROMPT: Final[str] = "Describe the image in detail."
