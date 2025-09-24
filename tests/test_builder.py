"""Tests for the dataset builder task."""

from __future__ import annotations

import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeObjectId(str):
    """Minimal stand-in for :class:`bson.ObjectId`."""

    def __new__(cls, value: str | None = None) -> "FakeObjectId":
        if value is None:
            value = "0" * 24
        return str.__new__(cls, value)


# Ensure ``tasks.builder`` can be imported without real ``bson``/``rstr`` deps.
sys.modules["bson"] = types.SimpleNamespace(ObjectId=FakeObjectId)
sys.modules["rstr"] = types.SimpleNamespace(xeger=lambda pattern: pattern)

from tasks.builder import DatasetBuilder


class FakeCollection:
    def __init__(self, documents: dict[FakeObjectId, dict]):
        self._documents = documents

    def find_one(self, query: dict) -> dict | None:
        return self._documents.get(query.get("_id"))


class FakeDB:
    def __init__(self, configurations: dict[FakeObjectId, dict]):
        self._configurations = configurations

    def get_collection(self, name: str) -> FakeCollection:
        if name == "models_configurations":
            return FakeCollection(self._configurations)
        if name == "models_data":
            return FakeCollection({})
        raise KeyError(name)


class DatasetBuilderOverlapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity_keys = ["VAT-SIREN_VAT_NUMBER", "VAT-SIREN_SIREN"]

    def test_nested_entity_inside_parent_is_preserved(self) -> None:
        nested_id = FakeObjectId("2" * 24)
        root_id = FakeObjectId("1" * 24)

        nested_config = {
            "_id": nested_id,
            "formats": ["{code}{VAT-SIREN_SIREN}"],
            "attributes": [
                {"key": "code", "value": "FR"},
                {"key": "VAT-SIREN_SIREN", "value": "123456789"},
            ],
        }

        root_config = {
            "_id": root_id,
            "formats": ["TVA {VAT-SIREN_VAT_NUMBER}"],
            "attributes": [
                {
                    "key": "VAT-SIREN_VAT_NUMBER",
                    "value": {
                        "rule": "configuration",
                        "type": "string",
                        "parameters": {"object_id": nested_id},
                    },
                }
            ],
        }

        db = FakeDB({nested_id: nested_config})

        builder = DatasetBuilder(
            configuration=root_config,
            db=db,
            entity_keys=self.entity_keys,
            randomizers=[],
        )

        sample = builder.generate_sample()

        self.assertEqual(sample["text"], "TVA FR123456789")
        self.assertEqual(
            sample["entities"],
            [
                [4, 15, "VAT-SIREN_VAT_NUMBER"],
                [6, 15, "VAT-SIREN_SIREN"],
            ],
        )

    def test_identical_spans_can_have_multiple_entities(self) -> None:
        nested_id = FakeObjectId("2" * 24)
        root_id = FakeObjectId("1" * 24)

        nested_config = {
            "_id": nested_id,
            "formats": ["{VAT-SIREN_SIREN}"],
            "attributes": [
                {"key": "VAT-SIREN_SIREN", "value": "123456789"},
            ],
        }

        root_config = {
            "_id": root_id,
            "formats": ["TVA {VAT-SIREN_VAT_NUMBER}"],
            "attributes": [
                {
                    "key": "VAT-SIREN_VAT_NUMBER",
                    "value": {
                        "rule": "configuration",
                        "type": "string",
                        "parameters": {"object_id": nested_id},
                    },
                }
            ],
        }

        db = FakeDB({nested_id: nested_config})

        builder = DatasetBuilder(
            configuration=root_config,
            db=db,
            entity_keys=self.entity_keys,
            randomizers=[],
        )

        sample = builder.generate_sample()

        self.assertEqual(sample["text"], "TVA 123456789")
        self.assertEqual(
            sample["entities"],
            [
                [4, 13, "VAT-SIREN_VAT_NUMBER"],
                [4, 13, "VAT-SIREN_SIREN"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
