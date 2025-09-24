import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _DummyDataset:
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_dict(cls, data):  # pragma: no cover - simple stub
        return cls(data)


class _DummyTokenizer:
    cls_token_id = 0

    def __init__(self):
        self._sequence_ids = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - stub
        return cls()

    def save_pretrained(self, *args, **kwargs):  # pragma: no cover - stub
        return None

    def __call__(
        self,
        questions,
        contexts,
        *,
        truncation,
        max_length,
        stride,
        return_overflowing_tokens,
        return_offsets_mapping,
        padding,
    ):
        batch = len(questions)
        input_ids = []
        attention_mask = []
        offset_mapping = []
        sequence_ids = []
        for question, context in zip(questions, contexts):
            # Very small deterministic tokenisation: every character is a token.
            q_tokens = [(0, 0)] * (len(question) + 2)
            c_offsets = []
            for index, _ in enumerate(context):
                c_offsets.append((index, index + 1))
            offsets = q_tokens + c_offsets + [(0, 0)]
            ids = list(range(len(offsets)))
            mask = [1] * len(offsets)
            sid = [0] * len(q_tokens) + [1] * len(c_offsets) + [0]
            input_ids.append(ids)
            attention_mask.append(mask)
            offset_mapping.append(offsets)
            sequence_ids.append(sid)
        self._sequence_ids = sequence_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "overflow_to_sample_mapping": list(range(batch)),
            "offset_mapping": offset_mapping,
        }

    def sequence_ids(self, index):  # pragma: no cover - stub
        return self._sequence_ids[index]


class _DummyModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - stub
        return cls()

    def save_pretrained(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyTrainer:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        self.kwargs = kwargs

    def train(self):  # pragma: no cover - stub
        return None

    def save_model(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _DummyTrainingArguments:
    def __init__(self, *args, **kwargs):  # pragma: no cover - stub
        self.kwargs = kwargs


sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset=_DummyDataset))
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(
        AutoTokenizer=_DummyTokenizer,
        AutoModelForQuestionAnswering=_DummyModel,
        DataCollatorWithPadding=lambda tokenizer: None,
        Trainer=_DummyTrainer,
        TrainingArguments=_DummyTrainingArguments,
    ),
)

from helpers.trainer import DocumentSchema, FieldPrediction, build_qa_examples


class DocumentSchemaTests(unittest.TestCase):
    def test_schema_extracts_fields_and_vocabulary(self) -> None:
        mapper = {
            "invoice": {
                "number": "INVOICE_NUMBER",
                "lines": [
                    {
                        "description": {
                            "label": "LINE_DESCRIPTION",
                            "synonyms": ["libellé"],
                        },
                        "quantity": "LINE_QUANTITY",
                    }
                ],
            },
            "address": {"city": "ADDRESS_CITY"},
        }

        schema = DocumentSchema.from_mapping(mapper, language="fr")

        self.assertIn("INVOICE_NUMBER", schema.labels_for_path(["invoice"]))
        self.assertIn("LINE_QUANTITY", schema.labels_for_path(["invoice", "lines"]))
        field = schema.get("LINE_DESCRIPTION")
        self.assertIsNotNone(field)
        assert field is not None
        self.assertIn("libellé", field.synonyms)
        self.assertTrue(field.cardinality == "multi")
        vocabulary = set(schema.vocabulary)
        self.assertIn("invoice", vocabulary)
        self.assertIn("number", vocabulary)
        self.assertIn("libellé", vocabulary)


class BuildQAExamplesTests(unittest.TestCase):
    def test_examples_created_for_entities(self) -> None:
        mapper = {
            "invoice": {
                "number": "INVOICE_NUMBER",
                "lines": [
                    {
                        "quantity": "LINE_QUANTITY",
                    }
                ],
            },
            "address": {"city": "ADDRESS_CITY"},
        }
        schema = DocumentSchema.from_mapping(mapper)

        text = "Facture F-123\nQuantité: 10\nVille: Paris"
        number_start = text.index("F-123")
        quantity_start = text.index("10")
        city_start = text.index("Paris")
        dataset = [
            {
                "_id": "doc1",
                "data": {
                    "text": text,
                    "entities": [
                        [number_start, number_start + 5, "INVOICE_NUMBER"],
                        [quantity_start, quantity_start + 2, "LINE_QUANTITY"],
                        [city_start, city_start + 5, "ADDRESS_CITY"],
                    ],
                },
            }
        ]

        examples = build_qa_examples(dataset, schema)

        labels = {example["label"] for example in examples}
        self.assertEqual(len(examples), 3)
        self.assertIn("INVOICE_NUMBER", labels)
        self.assertIn("LINE_QUANTITY", labels)
        self.assertIn("ADDRESS_CITY", labels)
        invoice_example = next(example for example in examples if example["label"] == "INVOICE_NUMBER")
        self.assertIn("Facture", invoice_example["context"])
        self.assertIn("valeur", invoice_example["question"].lower())


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        mapper = {
            "invoice": {
                "number": "INVOICE_NUMBER",
                "lines": [
                    {
                        "quantity": "LINE_QUANTITY",
                    }
                ],
            },
            "address": {"city": "ADDRESS_CITY"},
        }
        self.schema = DocumentSchema.from_mapping(mapper)

    def test_highest_confidence_is_kept(self) -> None:
        result = self.schema.aggregate_predictions(
            [
                FieldPrediction(label="INVOICE_NUMBER", value="F-123", confidence=0.91),
                FieldPrediction(label="INVOICE_NUMBER", value="F-124", confidence=0.5),
            ]
        )
        invoice = result["invoice"]["number"]
        self.assertEqual(invoice["value"], "F-123")
        self.assertAlmostEqual(invoice["confidence"], 0.91)

    def test_list_predictions_respect_positions(self) -> None:
        predictions = [
            FieldPrediction(
                label="LINE_QUANTITY",
                value="11",
                confidence=0.9,
                position=0,
            ),
            FieldPrediction(
                label="LINE_QUANTITY",
                value="8",
                confidence=0.65,
                position=1,
            ),
            FieldPrediction(
                label="ADDRESS_CITY",
                value="Paris",
                confidence=0.83,
            ),
        ]
        result = self.schema.aggregate_predictions(predictions)
        invoice = result["invoice"]
        lines = invoice["lines"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["quantity"]["value"], "11")
        self.assertEqual(lines[1]["quantity"]["confidence"], 0.65)
        self.assertEqual(result["address"]["city"]["value"], "Paris")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

