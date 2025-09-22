"""Callbacks utilities used during model fine-tuning."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from pymongo import MongoClient
from transformers import TrainerCallback


class MongoTrainLogger(TrainerCallback):
    """Push training metrics to MongoDB in real time."""

    def __init__(
        self,
        mongo_uri: str,
        dataset: str,
        job_coll: str = "datasets",
        *,
        model: str,
        version: str,
    ) -> None:
        if not mongo_uri:
            raise ValueError("mongo_uri must be provided to MongoTrainLogger")

        self.client = MongoClient(mongo_uri)
        self.collection = self.client.get_database()[job_coll]
        self.dataset_id = ObjectId(str(dataset))
        self.model = model
        self.version = version
        self.t0 = time.time()
        self.last_t = self.t0
        self.last_step = 0

    def _extract_numeric_metrics(self, logs: Optional[Dict[str, Any]]) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in (logs or {}).items()
            if isinstance(value, (int, float))
        }

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        now = time.time()
        step = state.global_step or 0
        dstep = step - self.last_step
        dt = max(1e-9, now - self.last_t)
        it_per_s = dstep / dt

        payload = {
            "ts": datetime.utcnow(),
            "step": step,
            "epoch": float(state.epoch or 0),
            "it_per_s": it_per_s,
            "seconds_from_start": now - self.t0,
            **self._extract_numeric_metrics(logs),
        }

        progress = None
        if state.max_steps and state.max_steps > 0:
            progress = step / state.max_steps

        self.collection.update_one(
            {"_id": self.dataset_id},
            {"$set": {"progress": progress, "last_log": payload}},
        )

        self.last_t = now
        self.last_step = step

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self.collection.update_one(
            {"_id": self.dataset_id},
            {"$set": {"finished_at": datetime.utcnow()}},
        )

    def close(self) -> None:
        self.client.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
