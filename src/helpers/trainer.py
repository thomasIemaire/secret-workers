from datasets import Dataset
import torch
import evaluate
import numpy as np
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

def prepare_dataset(examples, label_names):
    tok = CamembertTokenizerFast.from_pretrained("camembert-base")
    if "O" not in label_names:
        label_names = ["O"] + list(label_names)
    others = sorted([l for l in label_names if l != "O"])
    label_names = ["O"] + others
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    records = []
    for ex in examples:
        data = ex.get("data", {})
        text = (data.get("text") or "").strip()
        if not text:
            continue
        ents = data.get("entities", [])

        enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = enc["offset_mapping"]

        y = [label2id["O"]] * len(enc["input_ids"])
        for i, (s, e) in enumerate(offsets):
            if s == e == 0:
                y[i] = -100

        for start, end, lab in ents:
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue
            saw_begin = False
            for i, (s, e) in enumerate(offsets):
                if s == e == 0:
                    continue
                if e <= start or s >= end:
                    continue
                tag = f"B-{lab}" if not saw_begin and (s <= start < e) else f"I-{lab}"
                if tag in label2id:
                    y[i] = label2id[tag]
                    saw_begin = True

        if all(v == -100 for v in y):
            continue

        enc.pop("offset_mapping")
        enc["labels"] = [int(v) for v in y]
        records.append(enc)

    if not records:
        raise ValueError("Dataset vide après parsing")
    ds = Dataset.from_list(records)
    return ds, label2id, id2label

def compute_metrics(eval_pred, id2label):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=-1)
    true_preds, true_labels = [], []
    for p_seq, l_seq in zip(preds, labels):
        p_out, l_out = [], []
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            p_out.append(id2label[p])
            l_out.append(id2label[l])
        true_preds.append(p_out)
        true_labels.append(l_out)
    metric = evaluate.load("seqeval")
    return metric.compute(predictions=true_preds, references=true_labels)

def trainer(dataset, model, *, parameters=None, eval_dataset=None):
    parameters = parameters or {}
    label_names = model.get("labels", [])
    mname = model.get("name", "model")
    mversion = model.get("version", "1.0")
    train_ds, label2id, id2label = prepare_dataset(dataset, label_names)
    ner_model = CamembertForTokenClassification.from_pretrained(
        "camembert-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    tok = CamembertTokenizerFast.from_pretrained("camembert-base")
    collator = DataCollatorForTokenClassification(tok)
    use_fp16 = bool(parameters.get("fp16", False)) and torch.cuda.is_available()
    if parameters.get("fp16", False) and not torch.cuda.is_available():
        print("[trainer] fp16 demandé mais CUDA indisponible -> on désactive.", flush=True)
    out_dir = f"./src/models/{mname}/{mversion}"
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=parameters.get("learning_rate", 5e-5),
        per_device_train_batch_size=parameters.get("batch_size", 16),
        num_train_epochs=parameters.get("epochs", 5),
        weight_decay=parameters.get("weight_decay", 0.01),
        save_strategy="epoch",
        evaluation_strategy=("no" if eval_dataset is None else parameters.get("eval_strategy", "epoch")),
        logging_dir=f"./logs/{mname}/{mversion}",
        logging_steps=10,
        fp16=use_fp16,
        gradient_accumulation_steps=parameters.get("grad_accum", 1),
        group_by_length=True,
        dataloader_pin_memory=False
    )

    from collections import Counter
    c = Counter()
    for r in train_ds.select(range(min(50, len(train_ds)))):
        c.update(r["labels"])
    kept = sum(v for k, v in c.items() if k != -100)
    print(f"[trainer] first50 label counts={dict(c)} | kept_tokens={kept}", flush=True)

    tr = Trainer(
        model=ner_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None if eval_dataset is None else eval_dataset,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=(None if eval_dataset is None else (lambda p: compute_metrics(p, id2label))),
    )
    tr.train()
    tr.save_model(out_dir)
    tok.save_pretrained(out_dir)