import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np

from glados.robot.voice import AudioChunk, SpeakerWorker, VoiceWorker
from glados.core.conversation_store import ConversationStore


def test_voice_worker_synthesizes(mocker):
    """VoiceWorker reads TTS queue and puts audio into audio queue."""
    tts_q = queue.Queue()
    audio_q = queue.Queue()
    shutdown = threading.Event()

    tts_model = mocker.MagicMock()
    tts_model.sample_rate = 24000
    tts_model.generate_speech_audio.return_value = np.zeros(1000, dtype=np.float32)

    stc = mocker.MagicMock()
    stc.text_to_spoken.side_effect = lambda x: x

    tts_q.put("Привет")
    tts_q.put("<EOS>")

    worker = VoiceWorker(
        tts_queue=tts_q,
        audio_queue=audio_q,
        tts_model=tts_model,
        stc=stc,
        shutdown_event=shutdown,
    )

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(0.3)
    shutdown.set()
    t.join(timeout=2)

    assert audio_q.qsize() >= 1


def test_speaker_worker_plays_and_clears_flag():
    """SpeakerWorker should play audio and clear speaking flag on EOS."""
    audio_q = queue.Queue()
    speaking = threading.Event()
    processing = threading.Event()
    shutdown = threading.Event()
    conv = ConversationStore()

    mock_audio = MagicMock()
    sr = 22050

    worker = SpeakerWorker(
        audio_io=mock_audio,
        audio_queue=audio_q,
        conversation_store=conv,
        sample_rate=sr,
        shutdown_event=shutdown,
        speaking_event=speaking,
        processing_event=processing,
    )

    # Put audio chunk + EOS
    chunk = AudioChunk(audio=np.ones(1000, dtype=np.float32), text="test")
    eos = AudioChunk(audio=np.array([], dtype=np.float32), text="", is_eos=True)
    audio_q.put(chunk)
    audio_q.put(eos)

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(0.5)
    shutdown.set()
    t.join(timeout=2.0)

    mock_audio.start_speaking.assert_called_once()
    mock_audio.stop_speaking.assert_called()
    assert not speaking.is_set()
