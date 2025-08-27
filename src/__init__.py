import os, time, signal, traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
POLL_DELAY = float(os.getenv("POLL_DELAY", "2"))

client = MongoClient(MONGO_URI)
db = client.get_database()
col_tasks = db.get_collection("datasets")
col_data = db.get_collection("datasets_data")
col_models = db.get_collection("models")

from src.helpers.trainer import trainer as run_trainer

stop_event = Event()
def _stop(*_): stop_event.set()
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

def run_task(doc: dict):
    jid = str(doc["_id"])
    try:
        total = os.cpu_count() or 4
        per_worker = max(1, total // MAX_WORKERS)
        os.environ.setdefault("OMP_NUM_THREADS", str(per_worker))
        os.environ.setdefault("MKL_NUM_THREADS", str(per_worker))

        dataset = list(col_data.find({"dataset": ObjectId(doc["_id"])}))
        model_id = doc.get("model")
        if not model_id:
            raise ValueError("model_id manquant")
        model = col_models.find_one({"_id": ObjectId(model_id)})
        if not model:
            raise ValueError(f"modèle introuvable: {model_id}")
        
        version = doc.get("version", "1.0")
        params = doc.get("parameters", {})

        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "running", "started_at": datetime.utcnow()}})
        print(f"[{jid}] start training… model={model.get('name')} v={version} | items={len(dataset)}", flush=True)

        run_trainer(dataset, model, parameters=params, version=version)

        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "completed", "finished_at": datetime.utcnow()}})
        print(f"[{jid}] done ✅", flush=True)
    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed", "error": err, "finished_at": datetime.utcnow()}})
        print(f"[{jid}] failed ❌ {err}", flush=True)

def claim_one_pending():
    doc = col_tasks.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "claimed", "claimed_at": datetime.utcnow()}},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )
    if doc:
        print(f"Claimed task {doc['_id']}", flush=True)
    return doc

def main():
    print(f"Worker up (threads={MAX_WORKERS})", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = set()
        while not stop_event.is_set():
            futures = {f for f in futures if not f.done()}
            while not stop_event.is_set() and len(futures) < MAX_WORKERS:
                task = claim_one_pending()
                if not task:
                    break
                futures.add(executor.submit(run_task, task))
            time.sleep(POLL_DELAY if len(futures) < MAX_WORKERS else 0.2)