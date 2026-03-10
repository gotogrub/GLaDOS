import queue
import threading


def test_brain_priority_over_autonomy():
    """BrainWorker processes priority queue before autonomy."""
    from glados.robot.brain import BrainWorker

    priority_q = queue.Queue()
    autonomy_q = queue.Queue()
    tts_q = queue.Queue()
    shutdown = threading.Event()
    speaking = threading.Event()

    # Put items in both queues
    priority_q.put({"role": "user", "content": "hello", "_lane": "priority"})
    autonomy_q.put({"role": "user", "content": "scene changed", "_lane": "autonomy", "autonomy": True})

    worker = BrainWorker(
        priority_queue=priority_q,
        autonomy_queue=autonomy_q,
        tts_queue=tts_q,
        shutdown_event=shutdown,
        speaking_event=speaking,
        completion_url="http://localhost:11434/api/chat",
        model_name="gemma3:4b",
    )

    # _next_request should return priority first
    req = worker._next_request()
    assert req is not None
    assert req["content"] == "hello"

    req2 = worker._next_request()
    assert req2 is not None
    assert req2["content"] == "scene changed"
