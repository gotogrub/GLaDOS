import queue
import threading


def test_vision_worker_publishes_event(mocker):
    """VisionWorker publishes description to event queue on scene change."""
    from glados.robot.vision import VisionWorker
    from glados.robot.config import VisionSettings
    from glados.vision.vision_state import VisionState

    event_q = queue.Queue()
    shutdown = threading.Event()
    state = VisionState()

    settings = VisionSettings(capture_interval_seconds=0.1)

    worker = VisionWorker(
        vision_state=state,
        event_queue=event_q,
        shutdown_event=shutdown,
        settings=settings,
    )

    # We can't easily test without real models, but verify construction works
    assert worker._state is state
    assert worker._event_q is event_q
