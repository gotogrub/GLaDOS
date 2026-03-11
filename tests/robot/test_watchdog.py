import threading
import time

from glados.robot.watchdog import Watchdog


def test_watchdog_detects_dead_thread():
    """Watchdog should call on_failure when a thread dies."""
    failures = []

    def worker():
        raise RuntimeError("crash!")

    def on_failure(name: str, error: str):
        failures.append((name, error))

    t = threading.Thread(target=worker, name="TestWorker", daemon=True)
    t.start()
    t.join(timeout=1.0)

    wd = Watchdog(
        threads={"TestWorker": t},
        on_failure=on_failure,
        check_interval=0.1,
    )
    wd.check_once()
    assert len(failures) == 1
    assert failures[0][0] == "TestWorker"


def test_watchdog_ignores_alive_thread():
    """Watchdog should not trigger for alive threads."""
    failures = []
    stop = threading.Event()

    def worker():
        stop.wait()

    def on_failure(name, error):
        failures.append(name)

    t = threading.Thread(target=worker, name="AliveWorker", daemon=True)
    t.start()

    wd = Watchdog(
        threads={"AliveWorker": t},
        on_failure=on_failure,
        check_interval=0.1,
    )
    wd.check_once()
    stop.set()
    t.join(timeout=1.0)

    assert len(failures) == 0


def test_watchdog_notifies_only_once():
    """Watchdog should not re-notify for same dead thread."""
    failures = []

    def on_failure(name, error):
        failures.append(name)

    t = threading.Thread(target=lambda: None, name="DoneWorker", daemon=True)
    t.start()
    t.join(timeout=1.0)

    wd = Watchdog(
        threads={"DoneWorker": t},
        on_failure=on_failure,
    )
    wd.check_once()
    wd.check_once()
    assert len(failures) == 1
