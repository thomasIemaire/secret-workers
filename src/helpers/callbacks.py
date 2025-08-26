import time
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from transformers import TrainerCallback

class MongoTrainLogger(TrainerCallback):
    def __init__(self, mongo_uri: str, dataset: str, job_coll="datasets", *, model: str, version: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database()
        self.col_jobs = self.db[job_coll]
        self.dataset = dataset
        self.model = model
        self.version = version
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        step = state.global_step or 0
        dstep = step - self.last_step
        dt = max(1e-9, now - self.last_t)
        it_per_s = dstep / dt

        metrics = {}
        for k, v in (logs or {}).items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

        payload = {
            "ts": datetime.utcnow(),
            "step": step,
            "epoch": float(state.epoch or 0),
            "it_per_s": it_per_s,
            "seconds_from_start": now - self.t0,
            **metrics,
        }

        progress = None
        if state.max_steps and state.max_steps > 0:
            progress = step / state.max_steps
        self.col_jobs.update_one(
            {"_id": ObjectId(self.dataset)},
            {"$set": {"progress": progress, "last_log": payload}},
        )

        self.last_t = now
        self.last_step = step

    def on_train_end(self, args, state, control, **kwargs):
        self.col_jobs.update_one(
            {"_id": ObjectId(self.dataset)},
            {"$set": {
                "train_runtime": state.train_runtime,
                "train_samples_per_second": state.train_samples_per_second,
                "train_steps_per_second": state.train_steps_per_second,
                "train_loss": state.best_metric if state.best_metric is not None else None,
                "finished_at": datetime.utcnow(),
            }}
        )
