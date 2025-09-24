"""Tests for the trainer dataset preparation helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest


def _load_metric(_name: str):
    class _Metric:
        def compute(self, predictions, references):  # pragma: no cover - simple stub
            return {}

    return _Metric()


# Provide a lightweight stand-in for the optional ``evaluate`` dependency so that
# importing ``helpers.trainer`` during the tests does not require the real package.
sys.modules.setdefault("evaluate", types.SimpleNamespace(load=_load_metric))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))


class _DummyCollection:
    def update_one(self, *args, **kwargs):  # pragma: no cover - stub
        return None

    def find_one_and_update(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyDatabase:
    def __getitem__(self, _name):  # pragma: no cover - stub
        return _DummyCollection()

    def get_collection(self, _name):  # pragma: no cover - stub
        return _DummyCollection()


class _DummyMongoClient:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        pass

    def get_database(self):  # pragma: no cover - stub
        return _DummyDatabase()

    def close(self):  # pragma: no cover - stub
        return None


sys.modules.setdefault(
    "pymongo",
    types.SimpleNamespace(MongoClient=_DummyMongoClient, ReturnDocument=types.SimpleNamespace(AFTER="after")),
)
sys.modules.setdefault("bson", types.SimpleNamespace(ObjectId=lambda value: value))
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")


class _DummyNumpy(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(argmax=self._argmax, ndarray=list)

    @staticmethod
    def _argmax(array, axis=None):  # pragma: no cover - simple stub
        return 0


sys.modules.setdefault("numpy", _DummyNumpy())


class _DummyTensor:
    def __init__(self, data):
        self._data = data

    def numpy(self):  # pragma: no cover - simple stub
        return self._data


class _DummyTorch(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            tensor=lambda data: _DummyTensor(data),
            softmax=lambda tensor, dim=-1: tensor,
            qint8=object(),
            float16=object(),
            quantization=types.SimpleNamespace(quantize_dynamic=lambda model, modules, dtype: model),
            nn=types.SimpleNamespace(Linear=object),
        )


sys.modules.setdefault("torch", _DummyTorch())


class _DummyDataset:
    def __init__(self, records):
        self._records = list(records)

    @classmethod
    def from_list(cls, records):  # pragma: no cover - simple stub
        return cls(records)

    def __len__(self):  # pragma: no cover - simple stub
        return len(self._records)

    def __getitem__(self, index):  # pragma: no cover - simple stub
        return self._records[index]

    def select(self, indices):  # pragma: no cover - simple stub
        return _DummyDataset(self._records[i] for i in indices)

    def train_test_split(self, test_size=0.0, seed=None):  # pragma: no cover - simple stub
        return {"train": self, "test": self}


sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset=_DummyDataset))


sys.modules.setdefault(
    "seqeval.metrics",
    types.SimpleNamespace(
        classification_report=lambda *args, **kwargs: "",  # pragma: no cover - stub
        f1_score=lambda *args, **kwargs: 0.0,  # pragma: no cover - stub
        precision_score=lambda *args, **kwargs: 0.0,  # pragma: no cover - stub
        recall_score=lambda *args, **kwargs: 0.0,  # pragma: no cover - stub
    ),
)
sys.modules.setdefault("seqeval.scheme", types.SimpleNamespace(IOB2=object()))


class _DummyModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - stub
        return cls()

    def save_pretrained(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - stub
        return cls()

    def save_pretrained(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyTrainingArguments:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        self.kwargs = kwargs


class _DummyTrainer:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        pass

    def add_callback(self, _callback):  # pragma: no cover - stub
        return None

    def train(self):  # pragma: no cover - stub
        return None

    def save_model(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyEarlyStoppingCallback:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        self.kwargs = kwargs


class _DummySchedulerType(str):
    LINEAR = "linear"

    def __new__(cls, value="linear"):  # pragma: no cover - stub
        if value != cls.LINEAR:
            raise ValueError(value)
        return str.__new__(cls, value)


sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        CamembertForMaskedLM=_DummyModel,
        CamembertForTokenClassification=_DummyModel,
        CamembertTokenizerFast=_DummyTokenizer,
        DataCollatorForLanguageModeling=_DummyModel,
        DataCollatorForTokenClassification=_DummyModel,
        EarlyStoppingCallback=_DummyEarlyStoppingCallback,
        TrainerCallback=type("TrainerCallback", (), {}),
        SchedulerType=_DummySchedulerType,
        Trainer=_DummyTrainer,
        TrainingArguments=_DummyTrainingArguments,
    ),
)

from helpers.trainer import O_LABEL, prepare_dataset


class DummyTokenizer:
    """Tokenizer stub returning deterministic per-character offsets."""

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        truncation: bool,
        max_length: int,
    ) -> dict:
        if not return_offsets_mapping:
            raise ValueError("DummyTokenizer requires return_offsets_mapping=True")

        offsets = [(0, 0)]
        input_ids = [0]
        attention_mask = [1]

        for index, _ in enumerate(text):
            # Use one token per character to keep offsets simple and predictable.
            start = index
            end = index + 1
            offsets.append((start, end))
            input_ids.append(index + 1)
            attention_mask.append(1)

        # Trailing special token, mirroring fast tokenizers that include </s>.
        offsets.append((0, 0))
        input_ids.append(len(text) + 1)
        attention_mask.append(1)

        return {
            "input_ids": input_ids,
            "offset_mapping": offsets,
            "attention_mask": attention_mask,
        }


class PrepareDatasetOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.label_names = [
            O_LABEL,
            "B-PARENT",
            "I-PARENT",
            "B-CHILD",
            "I-CHILD",
        ]
        self.tokenizer = DummyTokenizer()

    def test_prepare_dataset_detects_overlapping_entities(self) -> None:
        overlapping = [
            {
                "data": {
                    "text": "TVA FR123456789",
                    "entities": [
                        [4, 15, "PARENT"],
                        [6, 15, "CHILD"],
                    ],
                }
            }
        ]

        with self.assertRaisesRegex(ValueError, "Les entités qui se chevauchent"):
            prepare_dataset(overlapping, self.label_names, self.tokenizer)

    def test_prepare_dataset_accepts_non_overlapping_entities(self) -> None:
        dataset = [
            {
                "data": {
                    "text": "SIREN 123456789",
                    "entities": [
                        [6, 15, "CHILD"],
                    ],
                }
            }
        ]

        prepared, label2id, _, _ = prepare_dataset(dataset, self.label_names, self.tokenizer)

        self.assertEqual(len(prepared), 1)
        # Ensure the correct tag is present in the encoded labels.
        child_label_id = label2id["B-CHILD"]
        flattened = list(prepared[0]["labels"])
        self.assertIn(child_label_id, flattened)


if __name__ == "__main__":
    unittest.main()
