"""Worker thread utilities."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

TaskClaimFn = Callable[[], Optional[dict]]
TaskRunFn = Callable[..., Any]


class Worker:
    """Manage a pool of worker threads dedicated to a specific task type."""

    def __init__(
        self,
        name: str,
        claim_fn: TaskClaimFn,
        run_fn: TaskRunFn,
        max_workers: int,
        poll_delay: float,
        stop_event: threading.Event,
        *,
        db: Any,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")

        self.name = name
        self.claim_fn = claim_fn
        self.run_fn = run_fn
        self.max_workers = max_workers
        self.poll_delay = poll_delay
        self.db = db
        self.stop_event = stop_event
        self._thread: Optional[threading.Thread] = None
        self._logger = logging.getLogger(f"worker.{name}")

    def _loop(self) -> None:
        self._logger.info("démarrage (threads=%s)", self.max_workers)
        futures: set[Future[Any]] = set()
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix=self.name
        ) as executor:
            while not self.stop_event.is_set():
                futures = {future for future in futures if not future.done()}

                made_progress = False
                while not self.stop_event.is_set() and len(futures) < self.max_workers:
                    task = self.claim_fn()
                    if not task:
                        break
                    futures.add(executor.submit(self._run_one, task))
                    made_progress = True

                if not made_progress:
                    timeout = (
                        self.poll_delay
                        if len(futures) < self.max_workers
                        else 0.2
                    )
                    self.stop_event.wait(timeout)

        self._logger.info("arrêt propre")

    def _run_one(self, task: dict) -> None:
        try:
            self.run_fn(doc=task, db=self.db, MAX_WORKERS=self.max_workers)
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception("échec de la tâche %s", task)

    def start(self) -> "Worker":
        self._thread = threading.Thread(
            target=self._loop, name=f"mgr-{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
