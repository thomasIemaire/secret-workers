"""Training utilities for sequence tagging models."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    CamembertForTokenClassification,
    CamembertTokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.helpers.callbacks import MongoTrainLogger

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "camembert-base"
MAX_SEQ_LENGTH = 512


def _normalise_labels(label_names: Sequence[str]) -> List[str]:
    unique = []
    seen = set()
    for label in label_names:
        if label not in seen:
            unique.append(label)
            seen.add(label)

    if "O" not in seen:
        unique.insert(0, "O")
        seen.add("O")

    others = sorted(label for label in unique if label != "O")
    return ["O", *others]


def _ensure_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def prepare_dataset(
    examples: Sequence[Mapping[str, Any]],
    label_names: Sequence[str],
    tokenizer: CamembertTokenizerFast,
) -> Tuple[Dataset, Dict[str, int], Dict[int, str]]:
    label_names = _normalise_labels(label_names)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    records: List[MutableMapping[str, Any]] = []
    for example in examples:
        data = example.get("data") or {}
        text = (data.get("text") or "").strip()
        if not text:
            continue

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        offsets = enc.pop("offset_mapping")

        labels = [label2id["O"]] * len(enc["input_ids"])
        for i, (start, end) in enumerate(offsets):
            if start == end == 0:
                labels[i] = -100

        entities = data.get("entities") or []
        for start, end, label in entities:
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or end <= start
            ):
                continue
            saw_begin = False
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end == 0:
                    continue
                if tok_end <= start or tok_start >= end:
                    continue
                tag = (
                    f"B-{label}"
                    if not saw_begin and (tok_start <= start < tok_end)
                    else f"I-{label}"
                )
                if tag in label2id:
                    labels[idx] = label2id[tag]
                    saw_begin = True

        if all(label == -100 for label in labels):
            continue

        enc["labels"] = [int(value) for value in labels]
        records.append(enc)

    if not records:
        raise ValueError("Dataset vide après parsing")

    return Dataset.from_list(records), label2id, id2label


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], id2label: Dict[int, str]):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_preds: List[List[str]] = []
    true_labels: List[List[str]] = []
    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds: List[str] = []
        seq_labels: List[str] = []
        for pred, label in zip(pred_seq, label_seq):
            if label == -100:
                continue
            seq_preds.append(id2label[int(pred)])
            seq_labels.append(id2label[int(label)])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    metric = evaluate.load("seqeval")
    return metric.compute(predictions=true_preds, references=true_labels)


def _collect_label_stats(dataset: Dataset) -> Counter:
    counter = Counter()
    sample_size = min(50, len(dataset))
    if sample_size:
        for record in dataset.select(range(sample_size)):
            counter.update(record["labels"])
    return counter


def trainer(
    dataset: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    eval_dataset: Optional[Dataset] = None,
    version: Optional[str] = None,
) -> Trainer:
    if not dataset:
        raise ValueError("Dataset vide: aucune donnée à entraîner")

    parameters = parameters or {}
    label_names = model.get("labels") or []
    model_reference = model.get("reference", "model")
    resolved_version = version or model.get("version", "1.0")

    tokenizer = CamembertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds, label2id, id2label = prepare_dataset(dataset, label_names, tokenizer)

    LOGGER.info(
        "Dataset prêt: %s exemples, %s étiquettes",
        len(train_ds),
        len(label2id),
    )

    ner_model = CamembertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    use_fp16 = _ensure_bool(parameters.get("fp16", False)) and torch.cuda.is_available()
    if parameters.get("fp16") and not torch.cuda.is_available():
        LOGGER.warning("fp16 demandé mais CUDA indisponible -> désactivation")

    output_dir = Path("sardine.agents") / model_reference / resolved_version
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path("logs") / model_reference / resolved_version
    logging_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=_ensure_float(parameters.get("learning_rate"), 5e-5),
        per_device_train_batch_size=_ensure_int(parameters.get("batch_size"), 16),
        num_train_epochs=_ensure_float(parameters.get("epochs"), 5),
        weight_decay=_ensure_float(parameters.get("weight_decay"), 0.01),
        save_strategy="epoch",
        evaluation_strategy=(
            "no"
            if eval_dataset is None
            else str(parameters.get("eval_strategy", "epoch"))
        ),
        logging_dir=str(logging_dir),
        logging_steps=_ensure_int(parameters.get("logging_steps", 10), 10),
        fp16=use_fp16,
        gradient_accumulation_steps=_ensure_int(parameters.get("grad_accum"), 1),
        group_by_length=True,
        dataloader_pin_memory=False,
    )

    label_stats = _collect_label_stats(train_ds)
    kept = sum(value for key, value in label_stats.items() if key != -100)
    LOGGER.info("Premiers comptes d'étiquettes (50 échantillons): %s", dict(label_stats))
    LOGGER.info("Tokens conservés: %s", kept)

    trainer_instance = Trainer(
        model=ner_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=(
            None
            if eval_dataset is None
            else (lambda predictions: compute_metrics(predictions, id2label))
        ),
    )

    mongo_uri = os.getenv("MONGO_URI")
    dataset_id = dataset[0].get("dataset") if dataset else None
    callback: Optional[MongoTrainLogger] = None
    if mongo_uri and dataset_id:
        callback = MongoTrainLogger(
            mongo_uri=mongo_uri,
            dataset=str(dataset_id),
            model=model.get("name", model_reference),
            version=resolved_version,
        )
        trainer_instance.add_callback(callback)

    trainer_instance.train()
    trainer_instance.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if callback is not None:
        callback.close()

    return trainer_instance
