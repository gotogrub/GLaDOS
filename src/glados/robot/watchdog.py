"""Thread watchdog with failure detection and logging."""
from __future__ import annotations

import threading
from typing import Callable

from loguru import logger


class Watchdog:
    """Monitors threads and calls on_failure when one dies."""

    def __init__(
        self,
        threads: dict[str, threading.Thread],
        on_failure: Callable[[str, str], None],
        check_interval: float = 2.0,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self._threads = threads
        self._on_failure = on_failure
        self._interval = check_interval
        self._shutdown = shutdown_event or threading.Event()
        self._notified: set[str] = set()

    def check_once(self) -> list[str]:
        """Check all threads once. Returns names of dead threads."""
        dead = []
        for name, thread in self._threads.items():
            if name in self._notified:
                continue
            if not thread.is_alive():
                logger.error("Watchdog: {} is DEAD", name)
                self._notified.add(name)
                self._on_failure(name, f"Thread {name} died")
                dead.append(name)
        return dead

    def run(self) -> None:
        """Run watchdog loop until shutdown."""
        logger.info("Watchdog started, monitoring {} threads.", len(self._threads))
        while not self._shutdown.is_set():
            self.check_once()
            self._shutdown.wait(timeout=self._interval)
        logger.info("Watchdog stopped.")
