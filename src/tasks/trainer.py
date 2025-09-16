from datetime import datetime
from bson import ObjectId
import os, traceback

from src.helpers.trainer import trainer as run_trainer

def run_task(*, doc: dict=None, db=None, MAX_WORKERS=2):
    if doc is None: return 
    
    col_tasks = db.get_collection("datasets")
    col_data = db.get_collection("datasets_data")
    col_models = db.get_collection("models")
    col_agents = db.get_collection("agents")

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

        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "training", "started_at": datetime.utcnow()}})
        print(f"[{jid}] start training… model={model.get('name')} v={version} | items={len(dataset)}", flush=True)

        run_trainer(dataset, model, parameters=params, version=version)

        col_data.delete_many({"dataset": ObjectId(doc["_id"])})

        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "completed", "finished_at": datetime.utcnow()}})

        path = f"sardine.agents/{doc.get('reference')}/{version}"

        col_agents.insert_one({
            "created_by": doc.get("created_by"),
            "created_at": datetime.utcnow(),
            "model": ObjectId(model_id),
            "version": version,
            "name": doc.get('name'),
            "reference": doc.get('reference'),
            "description": doc.get('description'),
            "path": path,
            "status": "enabled",
        })

        for repo in os.listdir(path):
            if repo.startswith('checkpoint-'):
                checkpoint_path = os.path.join(path, repo)
                if os.path.isdir(checkpoint_path):
                    os.remove(checkpoint_path)

        print(f"[{jid}] done ✅", flush=True)
    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        col_tasks.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed", "error": err, "finished_at": datetime.utcnow()}})
        print(f"[{jid}] failed ❌ {err}", flush=True)