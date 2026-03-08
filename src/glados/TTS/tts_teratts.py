from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizer:
    """Russian TTS synthesizer using TeraTTS (VITS) with ruaccent for stress placement.

    Uses the TeraTTS/glados2-g2p-vits model — a VITS model trained on Russian speech
    with a GLaDOS-like voice. Requires ``TeraTTS`` and ``ruaccent`` packages.

    Attributes:
        SAMPLE_RATE: Audio sample rate (22050 Hz, matching the TeraTTS model).
        DEFAULT_MODEL: HuggingFace model identifier.
    """

    SAMPLE_RATE: int = 22050
    DEFAULT_MODEL: str = "TeraTTS/glados2-g2p-vits"

    CUSTOM_STRESS: dict[str, str] = {
        "ГЛаДОС": "ГЛ+А+ДОС",
        "ГЛАДОС": "ГЛ+АДОС",
        "ГлаДОС": "Гл+аДОС",
        "ИИ": "+И-+И",
        "АИ": "+А-+И",
    }

    def __init__(self, model_name: str = DEFAULT_MODEL, length_scale: float = 1.1) -> None:
        from TeraTTS import TTS  # type: ignore[import-untyped]
        from ruaccent import RUAccent  # type: ignore[import-untyped]

        self.sample_rate = self.SAMPLE_RATE
        self.length_scale = length_scale

        self._tts = TTS(model_name, add_time_to_end=0.0, tokenizer_load_dict=True)

        self._accentizer = RUAccent()
        self._accentizer.load(
            omograph_model_size="turbo",
            use_dictionary=True,
            custom_dict=self.CUSTOM_STRESS,
        )

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Convert Russian text to speech audio.

        Applies stress placement via ruaccent, then synthesizes with TeraTTS.

        Parameters:
            text: Russian text to synthesize.

        Returns:
            Audio samples as float32 numpy array at 22050 Hz.
        """
        processed = self._accentizer.process_all(text.strip())
        audio: NDArray[np.float32] = self._tts(processed, play=False, length_scale=self.length_scale)
        return np.array(audio, dtype=np.float32)
