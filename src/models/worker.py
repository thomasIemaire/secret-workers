import threading
import logging
from concurrent.futures import ThreadPoolExecutor

class Worker:
    def __init__(self, name, claim_fn, run_fn, max_workers, poll_delay, stop_event: threading.Event, *, db):
        self.name = name
        self.claim_fn = claim_fn
        self.run_fn = run_fn
        self.max_workers = max_workers
        self.poll_delay = poll_delay
        self.db = db
        self.stop_event = stop_event
        self._thread = None

    def _loop(self):
        logging.info(f"{self.name}: démarrage (threads={self.max_workers})")
        futures = set()
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.name) as executor:
            while not self.stop_event.is_set():
                # Nettoyage des futures terminées
                futures = {f for f in futures if not f.done()}

                # Remplir le pool
                made_progress = False
                while not self.stop_event.is_set() and len(futures) < self.max_workers:
                    task = self.claim_fn()
                    if not task:
                        break
                    futures.add(executor.submit(self._run_one, task))
                    made_progress = True

                # Eviter le busy-wait : attendre avec timeout
                if not made_progress:
                    # si pas de tâche dispo -> attendre poll_delay, sinon on est à capacité -> petite attente
                    timeout = self.poll_delay if len(futures) < self.max_workers else 0.2
                    self.stop_event.wait(timeout)

        logging.info(f"{self.name}: arrêt propre")

    def _run_one(self, task):
        try:
            return self.run_fn(doc=task, db=self.db, MAX_WORKERS=self.max_workers)
        except Exception as e:
            logging.exception(f"{self.name}: échec de la tâche {task}: {e}")

    def start(self):
        self._thread = threading.Thread(target=self._loop, name=f"mgr-{self.name}", daemon=True)
        self._thread.start()
        return self

    def join(self):
        if self._thread is not None:
            self._thread.join()
