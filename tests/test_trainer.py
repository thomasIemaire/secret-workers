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

from helpers.document import (
    CompositeAgent,
    DocumentExtractionAgent,
    DocumentSchema,
    DocumentVocabulary,
)
from helpers.trainer import O_LABEL, prepare_dataset


class DummyTokenizer:
    """Tokenizer stub returning deterministic per-character offsets."""

    def __init__(self) -> None:
        self.added_tokens: list[str] = []

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

    def add_tokens(self, tokens) -> int:  # pragma: no cover - simple stub
        token_list = list(tokens)
        if not token_list:
            return 0
        self.added_tokens.extend(token_list)
        return len(token_list)


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


class DocumentSchemaMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        schema_mapping = {
            "fields": [
                {
                    "name": "invoice_number",
                    "label": "INVOICE_NUMBER",
                    "path": ["invoice", "number"],
                },
                {
                    "name": "customer_name",
                    "label": "CUSTOMER_NAME",
                    "path": ["customer", "name"],
                },
            ],
            "collections": [
                {
                    "name": "lines",
                    "path": ["invoice", "lines"],
                    "fields": [
                        {"name": "description", "label": "LINE_DESCRIPTION", "path": ["description"]},
                        {"name": "quantity", "label": "LINE_QUANTITY", "path": ["quantity"]},
                    ],
                }
            ],
        }
        self.schema = DocumentSchema.from_mapping(schema_mapping)
        self.agent = DocumentExtractionAgent(name="invoice", schema=self.schema, threshold=0.5)

    def test_schema_maps_predictions_to_json_structure(self) -> None:
        predictions = [
            {"label": "INVOICE_NUMBER", "text": "F2024-001", "score": 0.92, "start": 0, "end": 9},
            {"label": "CUSTOMER_NAME", "text": "ACME", "score": 0.9, "start": 10, "end": 14},
            {"label": "LINE_DESCRIPTION", "text": "Service A", "score": 0.85, "start": 20, "end": 29, "group": 0},
            {"label": "LINE_QUANTITY", "text": "2", "score": 0.8, "start": 30, "end": 31, "group": 0},
            {"label": "LINE_DESCRIPTION", "text": "Service B", "score": 0.88, "start": 40, "end": 49, "group": 1},
        ]

        result = self.agent.map_predictions(predictions)

        self.assertEqual(result["invoice"]["number"]["value"], "F2024-001")
        self.assertGreater(result["invoice"]["number"]["confidence"], 0.9)
        self.assertEqual(result["customer"]["name"]["value"], "ACME")

        lines = result["invoice"]["lines"]
        self.assertEqual(len(lines), 2)
        first_line = lines[0]
        self.assertEqual(first_line["description"]["value"], "Service A")
        self.assertEqual(first_line["quantity"]["value"], "2")
        self.assertAlmostEqual(first_line["_confidence"], (0.85 + 0.8) / 2, places=4)
        self.assertEqual(first_line["_row_id"], 0)
        self.assertEqual(first_line["description"]["provenance"]["metadata"]["group"], 0)

        second_line = lines[1]
        self.assertEqual(second_line["description"]["value"], "Service B")
        self.assertIsNone(second_line["quantity"]["value"])
        self.assertAlmostEqual(second_line["_confidence"], 0.88, places=4)


class DocumentVocabularyTests(unittest.TestCase):
    def test_vocabulary_includes_domain_terms_and_applies_to_tokenizer(self) -> None:
        examples = [
            {"data": {"text": "Facture adressée à Mme Dupont 75001 Paris"}},
            {"data": {"text": "Total facture 1200 EUR payé par virement bancaire"}},
        ]
        vocabulary = DocumentVocabulary.from_examples(examples, additional_terms=["référence"])
        metadata = vocabulary.to_metadata()

        self.assertIn("facture", metadata["tokens"])
        self.assertIn("référence", metadata["tokens"])
        self.assertGreaterEqual(metadata["size"], len(DocumentVocabulary.DEFAULT_TERMS))

        tokenizer = DummyTokenizer()
        added = vocabulary.apply_to_tokenizer(tokenizer)
        self.assertEqual(added, len(vocabulary.tokens))
        self.assertEqual(tokenizer.added_tokens, vocabulary.tokens)


class CompositeAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        address_schema = DocumentSchema.from_mapping(
            {
                "fields": [
                    {
                        "name": "company_address",
                        "label": "ADDRESS",
                        "path": ["company", "address"],
                    }
                ]
            }
        )
        lines_schema = DocumentSchema.from_mapping(
            {
                "collections": [
                    {
                        "name": "lines",
                        "path": ["invoice", "lines"],
                        "fields": [
                            {"name": "description", "label": "LINE_DESCRIPTION", "path": ["description"]},
                            {"name": "quantity", "label": "LINE_QUANTITY", "path": ["quantity"]},
                        ],
                    }
                ]
            }
        )
        self.address_agent = DocumentExtractionAgent(name="adresse", schema=address_schema, threshold=0.4)
        self.lines_agent = DocumentExtractionAgent(name="lignes", schema=lines_schema, threshold=0.4)
        self.composite = CompositeAgent([self.address_agent, self.lines_agent])

    def test_composite_merges_outputs_using_best_confidence(self) -> None:
        predictions = {
            "adresse": [
                {"label": "ADDRESS", "text": "12 rue Bleue", "score": 0.6, "start": 0, "end": 12},
                {"label": "ADDRESS", "text": "14 rue Verte", "score": 0.95, "start": 0, "end": 12},
            ],
            "lignes": [
                {"label": "LINE_DESCRIPTION", "text": "Produit A", "score": 0.7, "start": 20, "end": 29, "group": 0},
                {"label": "LINE_QUANTITY", "text": "5", "score": 0.65, "start": 30, "end": 31, "group": 0},
            ],
        }

        result = self.composite.combine(predictions)

        self.assertEqual(result["company"]["address"]["value"], "14 rue Verte")
        lines = result["invoice"]["lines"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["description"]["value"], "Produit A")
        self.assertEqual(lines[0]["quantity"]["value"], "5")
        self.assertAlmostEqual(lines[0]["_confidence"], (0.7 + 0.65) / 2, places=4)


if __name__ == "__main__":
    unittest.main()
