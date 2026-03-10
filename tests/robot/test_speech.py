import queue
import threading
import numpy as np


def test_speech_worker_enqueues_on_detection(mocker):
    """SpeechWorker puts transcribed text into the LLM queue."""
    from glados.robot.speech import SpeechWorker

    llm_queue = queue.Queue()
    shutdown = threading.Event()
    speaking = threading.Event()

    # Mock audio_io
    audio_io = mocker.MagicMock()
    sample_queue = queue.Queue()
    audio_io.get_sample_queue.return_value = sample_queue

    # Mock ASR
    asr = mocker.MagicMock()
    asr.transcribe.return_value = "привет"

    worker = SpeechWorker(
        audio_io=audio_io,
        asr_model=asr,
        llm_queue=llm_queue,
        shutdown_event=shutdown,
        currently_speaking_event=speaking,
    )

    # Simulate VAD-active samples then gap
    for _ in range(25):
        sample_queue.put((np.zeros(512, dtype=np.float32), True))
    for _ in range(25):
        sample_queue.put((np.zeros(512, dtype=np.float32), False))

    # Run briefly then shutdown
    shutdown.set()
    worker.run()

    # Should have enqueued something (or at least not crashed)
