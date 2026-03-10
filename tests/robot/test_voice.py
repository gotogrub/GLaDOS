import queue
import threading
import time
import numpy as np


def test_voice_worker_synthesizes(mocker):
    """VoiceWorker reads TTS queue and puts audio into audio queue."""
    from glados.robot.voice import VoiceWorker

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

    # Run in a thread, let it process, then shutdown
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(0.3)
    shutdown.set()
    t.join(timeout=2)

    assert audio_q.qsize() >= 1
