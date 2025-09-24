"""Training utilities for LayoutLM-based document extraction agents.

This module restructures the training helpers around a document question
answering workflow.  The goal is to make it easy to fine-tune a
pretrained LayoutLM model on document datasets, expose a document oriented
vocabulary, and aggregate predictions into JSON structures with
confidence scores.  Agents that specialise on a subset of fields can be
trained independently and their outputs later merged together.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "impira/layoutlm-document-qa"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
TERMINAL_KEYS = {
    "label",
    "question",
    "vocabulary",
    "synonyms",
    "cardinality",
    "position",
    "language",
}


@dataclass(frozen=True)
class PathSegment:
    """Represents a segment in a JSON path."""

    name: str
    kind: str = "object"  # "object", "list", or "field"

    def __post_init__(self) -> None:
        if self.kind not in {"object", "list", "field"}:
            raise ValueError(f"Type de segment inconnu: {self.kind}")


@dataclass(frozen=True)
class FieldSpec:
    """Metadata describing a field that should be extracted."""

    label: str
    path: Tuple[PathSegment, ...]
    question: str
    vocabulary: Tuple[str, ...]
    cardinality: str = "single"
    synonyms: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Chaque champ doit avoir un identifiant")
        if not self.path:
            raise ValueError("Un champ doit être associé à un chemin JSON")
        if self.cardinality not in {"single", "multi"}:
            raise ValueError("Cardinalité inconnue: %s" % self.cardinality)

    @property
    def container(self) -> Tuple[PathSegment, ...]:
        """Return the path leading to the parent container."""

        return self.path[:-1]

    @property
    def leaf(self) -> PathSegment:
        return self.path[-1]

    def with_path(self, new_path: Sequence[PathSegment]) -> "FieldSpec":
        return FieldSpec(
            label=self.label,
            path=tuple(new_path),
            question=self.question,
            vocabulary=self.vocabulary,
            cardinality=self.cardinality,
            synonyms=self.synonyms,
        )


@dataclass(frozen=True)
class FieldPrediction:
    """Prediction returned by a specialised agent."""

    label: str
    value: str
    confidence: float
    position: Optional[int] = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        confidence = 0.0 if self.confidence is None else float(self.confidence)
        confidence = max(0.0, min(1.0, confidence))
        object.__setattr__(self, "confidence", confidence)
        if self.position is not None and self.position < 0:
            object.__setattr__(self, "position", 0)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FieldPrediction":
        return cls(
            label=str(payload.get("label")),
            value=str(payload.get("value", "")),
            confidence=float(payload.get("confidence", 0.0)),
            position=payload.get("position"),
            metadata=payload.get("metadata"),
        )


class DocumentSchema:
    """Schema describing how to extract fields from a document."""

    def __init__(self, fields: Sequence[FieldSpec], *, language: str = "fr") -> None:
        if not fields:
            raise ValueError("Le mapper du modèle ne contient aucun champ")
        self.language = language
        self.fields: Tuple[FieldSpec, ...] = tuple(fields)
        self._field_by_label: Dict[str, FieldSpec] = {
            field.label: field for field in self.fields
        }
        vocabulary: set[str] = set()
        for field in self.fields:
            vocabulary.update(field.vocabulary)
        self.vocabulary: Tuple[str, ...] = tuple(sorted(vocabulary))

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        language: str = "fr",
    ) -> "DocumentSchema":
        fields = list(_collect_fields(mapping, language=language))
        return cls(fields, language=language)

    def get(self, label: str) -> Optional[FieldSpec]:
        return self._field_by_label.get(label)

    def labels_for_path(self, path: Sequence[str]) -> List[str]:
        prefix = tuple(path)
        matches: List[str] = []
        for field in self.fields:
            container_names = tuple(segment.name for segment in field.container)
            if container_names[: len(prefix)] == prefix:
                matches.append(field.label)
        return matches

    def as_mapping(self) -> Dict[str, Any]:
        root: Dict[str, Any] = {}
        for field in self.fields:
            cursor: MutableMapping[str, Any] | List[Any]
            cursor = root
            for segment in field.container:
                if segment.kind == "object":
                    cursor = cursor.setdefault(segment.name, {})  # type: ignore[assignment]
                elif segment.kind == "list":
                    items = cursor.setdefault(segment.name, [{}])  # type: ignore[assignment]
                    if not items:
                        items.append({})
                    cursor = items[0]  # type: ignore[index]
            leaf_name = field.leaf.name
            if isinstance(cursor, list):
                cursor = cursor[0]  # pragma: no cover - defensive
            cursor[leaf_name] = field.label  # type: ignore[index]
        return root

    def aggregate_predictions(
        self, *prediction_groups: Iterable[FieldPrediction] | FieldPrediction
    ) -> Dict[str, Any]:
        flat_predictions: List[FieldPrediction] = []
        for group in prediction_groups:
            if group is None:
                continue
            if isinstance(group, FieldPrediction):
                flat_predictions.append(group)
                continue
            for prediction in group:
                if isinstance(prediction, FieldPrediction):
                    flat_predictions.append(prediction)
                elif isinstance(prediction, Mapping):
                    flat_predictions.append(FieldPrediction.from_mapping(prediction))
                else:  # pragma: no cover - defensive
                    raise TypeError(
                        "Predictions must be FieldPrediction instances or mappings"
                    )

        flat_predictions.sort(key=lambda pred: pred.confidence, reverse=True)

        result: Dict[str, Any] = {}

        for prediction in flat_predictions:
            field = self.get(prediction.label)
            if field is None:
                LOGGER.debug("Étiquette inconnue ignorée: %s", prediction.label)
                continue
            if not prediction.value:
                continue

            cursor: Any = result
            list_cursors: List[Tuple[List[Any], int]] = []

            for segment in field.container:
                if segment.kind == "object":
                    cursor = cursor.setdefault(segment.name, {})
                elif segment.kind == "list":
                    position = prediction.position
                    if not isinstance(cursor.get(segment.name), list):
                        cursor[segment.name] = []
                    cursor_list = cursor[segment.name]
                    if position is None:
                        position = len(cursor_list)
                    while len(cursor_list) <= position:
                        cursor_list.append({})
                    list_cursors.append((cursor_list, position))
                    cursor = cursor_list[position]

            leaf_name = field.leaf.name
            existing = cursor.get(leaf_name)
            if isinstance(existing, Mapping) and existing.get("confidence", -1) >= prediction.confidence:
                continue
            cursor[leaf_name] = {
                "value": prediction.value,
                "confidence": prediction.confidence,
            }

            for cursor_list, position in list_cursors:
                if position is None:
                    continue
                cursor_list[position] = _prune_empty(cursor_list[position]) or {}

        return _prune_empty(result) or {}


def _tokenize_identifier(identifier: str) -> List[str]:
    tokens = re.split(r"[^\w]+", identifier, flags=re.UNICODE)
    return [token.lower() for token in tokens if token]


def _humanise_identifier(identifier: str) -> str:
    tokens = _tokenize_identifier(identifier)
    if not tokens:
        return identifier.lower()
    return " ".join(tokens)


def _generate_question(path: Sequence[PathSegment], *, language: str) -> str:
    field_name = _humanise_identifier(path[-1].name)
    context = " ".join(_humanise_identifier(segment.name) for segment in path[:-1])
    if language.lower().startswith("fr"):
        if context:
            return f"Quelle est la valeur de {field_name} dans {context} ?"
        return f"Quelle est la valeur de {field_name} ?"
    if context:
        return f"What is the value of {field_name} in {context}?"
    return f"What is the value of {field_name}?"


def _collect_vocabulary(
    path: Sequence[PathSegment],
    *,
    label: str,
    synonyms: Sequence[str],
) -> Tuple[str, ...]:
    vocab: set[str] = set()
    for segment in path:
        vocab.update(_tokenize_identifier(segment.name))
    vocab.update(_tokenize_identifier(label))
    for synonym in synonyms:
        vocab.update(_tokenize_identifier(str(synonym)))
    return tuple(sorted(vocab))


def _is_terminal_mapping(value: Mapping[str, Any]) -> bool:
    return any(key in TERMINAL_KEYS for key in value.keys())


def _build_field_spec(
    *,
    prefix: Tuple[PathSegment, ...],
    key: str,
    value: Any,
    language: str,
) -> FieldSpec:
    if isinstance(value, Mapping) and _is_terminal_mapping(value):
        label = str(value.get("label"))
        question = value.get("question")
        synonyms = tuple(value.get("synonyms") or ())
        vocabulary_override = value.get("vocabulary")
        cardinality = value.get("cardinality") or "multi" if any(
            segment.kind == "list" for segment in prefix
        ) else "single"
    else:
        label = str(value)
        question = None
        synonyms = ()
        vocabulary_override = None
        cardinality = "multi" if any(segment.kind == "list" for segment in prefix) else "single"

    path = prefix + (PathSegment(name=key, kind="field"),)
    resolved_question = question or _generate_question(path, language=language)
    if vocabulary_override:
        vocabulary = tuple({str(token).lower() for token in vocabulary_override})
    else:
        vocabulary = _collect_vocabulary(path, label=label, synonyms=synonyms)
    return FieldSpec(
        label=label,
        path=path,
        question=resolved_question,
        vocabulary=vocabulary,
        cardinality=cardinality,
        synonyms=synonyms,
    )


def _collect_fields(
    mapping: Mapping[str, Any],
    *,
    language: str,
    prefix: Tuple[PathSegment, ...] = (),
) -> Iterator[FieldSpec]:
    for key, value in mapping.items():
        if isinstance(value, Mapping) and not _is_terminal_mapping(value):
            new_prefix = prefix + (PathSegment(name=str(key), kind="object"),)
            yield from _collect_fields(value, language=language, prefix=new_prefix)
            continue
        if isinstance(value, list):
            if not value:
                continue
            first = value[0]
            new_prefix = prefix + (PathSegment(name=str(key), kind="list"),)
            if isinstance(first, Mapping):
                yield from _collect_fields(first, language=language, prefix=new_prefix)
            else:
                yield _build_field_spec(
                    prefix=new_prefix,
                    key=str(key),
                    value=first,
                    language=language,
                )
            continue
        yield _build_field_spec(
            prefix=prefix,
            key=str(key),
            value=value,
            language=language,
        )


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned: Dict[str, Any] = {}
        for key, child in value.items():
            child_pruned = _prune_empty(child)
            if child_pruned is not None:
                pruned[key] = child_pruned
        return pruned or None
    if isinstance(value, list):
        new_list = []
        for element in value:
            child_pruned = _prune_empty(element)
            if child_pruned is not None:
                new_list.append(child_pruned)
        return new_list or None
    return value


def build_qa_examples(
    dataset: Sequence[Mapping[str, Any]],
    schema: DocumentSchema,
) -> List[Dict[str, Any]]:
    qa_examples: List[Dict[str, Any]] = []
    for index, record in enumerate(dataset):
        payload = record.get("data") or {}
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        entities = payload.get("entities") or []
        for entity_index, raw_entity in enumerate(entities):
            try:
                start, end, label = raw_entity
            except (TypeError, ValueError):
                continue
            field = schema.get(str(label))
            if field is None:
                continue
            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                continue
            start = max(0, start)
            end = min(len(text), end)
            if end <= start:
                continue
            answer_text = text[start:end]
            qa_examples.append(
                {
                    "id": f"{record.get('_id', index)}::{label}::{entity_index}",
                    "question": field.question,
                    "context": text,
                    "answers": {"text": [answer_text], "answer_start": [start]},
                    "label": str(label),
                    "path": [segment.name for segment in field.path],
                }
            )
    return qa_examples


def _tokenize_for_qa(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    *,
    max_length: int,
    stride: int,
) -> Dataset:
    questions = [example["question"] for example in examples]
    contexts = [example["context"] for example in examples]
    answers = [example["answers"] for example in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []
    example_ids: List[str] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        example_ids.append(examples[sample_index]["id"])

        if not answer["text"]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer["answer_start"][0]
        answer_text = answer["text"][0]
        end_char = start_char + len(answer_text)

        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(sequence_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if (
            token_start_index >= len(offsets)
            or token_end_index < 0
            or offsets[token_start_index][0] > start_char
            or offsets[token_end_index][1] < end_char
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while (
            token_start_index < len(offsets)
            and offsets[token_start_index][0] <= start_char
        ):
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char and token_end_index >= 0:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["example_id"] = example_ids
    return Dataset.from_dict(tokenized)


def prepare_qa_dataset(
    dataset: Sequence[Mapping[str, Any]],
    *,
    schema: DocumentSchema,
    tokenizer,
    max_length: Optional[int] = None,
    doc_stride: Optional[int] = None,
) -> Tuple[Dataset, Dict[str, Any]]:
    qa_examples = build_qa_examples(dataset, schema)
    if not qa_examples:
        raise ValueError("Dataset vide: aucun exemple question/réponse généré")
    max_len = max_length or MAX_SEQ_LENGTH
    stride = doc_stride or DOC_STRIDE
    tokenized = _tokenize_for_qa(qa_examples, tokenizer, max_length=max_len, stride=stride)
    metadata = {
        "example_count": len(qa_examples),
        "schema": schema.as_mapping(),
        "vocabulary": schema.vocabulary,
    }
    return tokenized, metadata


def trainer(
    dataset: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    eval_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
    version: Optional[str] = None,
) -> Trainer:
    if not dataset:
        raise ValueError("Dataset vide: aucune donnée à entraîner")

    parameters = dict(parameters or {})
    mapper = model.get("mapper")
    if not isinstance(mapper, Mapping):
        raise ValueError("Le modèle doit fournir un mapper de champs")

    language = str(parameters.get("language", "fr"))
    schema = DocumentSchema.from_mapping(mapper, language=language)

    base_model = parameters.get("base_model") or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset, metadata = prepare_qa_dataset(
        dataset,
        schema=schema,
        tokenizer=tokenizer,
        max_length=parameters.get("max_length"),
        doc_stride=parameters.get("doc_stride"),
    )

    if eval_dataset:
        eval_encoded, _ = prepare_qa_dataset(
            eval_dataset,
            schema=schema,
            tokenizer=tokenizer,
            max_length=parameters.get("max_length"),
            doc_stride=parameters.get("doc_stride"),
        )
    else:
        eval_encoded = None

    output_dir = parameters.get("output_dir", "sardine.agents/tmp")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(parameters.get("batch_size", 2)),
        num_train_epochs=float(parameters.get("epochs", 3)),
        learning_rate=float(parameters.get("learning_rate", 3e-5)),
        logging_steps=int(parameters.get("logging_steps", 10)),
        evaluation_strategy="no" if eval_encoded is None else "epoch",
        save_total_limit=int(parameters.get("save_total_limit", 2)),
        save_strategy="epoch",
        fp16=bool(parameters.get("fp16", False)),
    )

    qa_model = AutoModelForQuestionAnswering.from_pretrained(base_model)

    collator = DataCollatorWithPadding(tokenizer)

    trainer_instance = Trainer(
        model=qa_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_encoded,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    LOGGER.info(
        "Entraînement LayoutLM: %s exemples, vocabulaire=%s",
        metadata["example_count"],
        len(metadata["vocabulary"]),
    )

    trainer_instance.train()
    trainer_instance.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer_instance


__all__ = [
    "DocumentSchema",
    "FieldPrediction",
    "FieldSpec",
    "PathSegment",
    "build_qa_examples",
    "prepare_qa_dataset",
    "trainer",
]

