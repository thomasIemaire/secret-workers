import os, time, signal
from threading import Event
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv

from src.tasks.builder import run_task as run_builder
from src.tasks.trainer import run_task as run_trainer

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MAX_WORKERS_BUILDER = int(os.getenv("MAX_WORKERS_BUILDER", "5"))
MAX_WORKERS_TRAINER = int(os.getenv("MAX_WORKERS_TRAINER", "2"))
POLL_DELAY = float(os.getenv("POLL_DELAY", "2"))

db = MongoClient(MONGO_URI).get_database()

stop_event = Event()
def _stop(*_): stop_event.set()
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

def claim_one_status(status: str, new_status: str):
    return db.get_collection("datasets").find_one_and_update(
        {"status": status},
        {"$set": {"status": new_status}},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )

import logging
from src.models.worker import Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s")

def main():
    logging.info("Lancement des workers…")

    workers = [
        Worker(
            "builder",
            lambda: claim_one_status("empty", "generating"),
            run_builder,
            MAX_WORKERS_BUILDER,
            POLL_DELAY,
            stop_event,
            db=db,
        ).start(),
        Worker(
            "trainer",
            lambda: claim_one_status("ready", "training"),
            run_trainer,
            MAX_WORKERS_TRAINER,
            POLL_DELAY,
            stop_event,
            db=db,
        ).start(),
    ]

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Signal reçu, arrêt en cours…")
        stop_event.set()
        for w in workers:
            w.join()
        logging.info("Arrêt terminé.")