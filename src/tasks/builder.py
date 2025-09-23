"""Dataset generation task."""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rstr
from bson import ObjectId

LOGGER = logging.getLogger(__name__)
PLACEHOLDER_PATTERN = re.compile(r"\{(?P<key>[^:{}]+)(?::[^{}]*)?\}")
BULK_INSERT_SIZE = 500


def run_task(*, doc: Optional[Mapping[str, Any]] = None, db=None, MAX_WORKERS: int = 2) -> None:
    if not doc or db is None:
        LOGGER.warning("builder: tâche ignorée (doc ou db manquant)")
        return

    datasets = db.get_collection("datasets")
    data_collection = db.get_collection("datasets_data")
    models = db.get_collection("models")
    configs = db.get_collection("models_configurations")

    dataset_id = doc["_id"]
    if not isinstance(dataset_id, ObjectId):
        dataset_id = ObjectId(dataset_id)

    model_id = doc.get("model")
    if not model_id:
        raise ValueError("Model configuration is missing")

    if not isinstance(model_id, ObjectId):
        model_id = ObjectId(model_id)

    model = models.find_one({"_id": model_id})
    if not model:
        raise ValueError(f"Model introuvable: {model_id}")

    configuration_id = model.get("configuration")
    if not configuration_id:
        raise ValueError("Model configuration is missing")

    if not isinstance(configuration_id, ObjectId):
        configuration_id = ObjectId(configuration_id)

    configuration = configs.find_one({"_id": configuration_id})
    if not configuration:
        raise ValueError(f"Configuration introuvable: {configuration_id}")

    entity_keys = list((model.get("entities") or {}).keys())
    randomizers = model.get("randomizers") or []
    builder = DatasetBuilder(configuration=configuration, db=db, entity_keys=entity_keys, randomizers=randomizers)

    size_info = doc.get("size", {})
    max_possibilities = int(configuration.get("possibilities", 1e5))
    formats_count = len(configuration.get("formats") or [])
    dataset_size = determine_dataset_size(size_info, max_possibilities, formats_count)

    LOGGER.info("builder[%s]: génération de %s entrées", dataset_id, dataset_size)
    datasets.update_one(
        {"_id": dataset_id},
        {"$set": {"status": "generating", "progress": 0.0}},
    )

    samples: List[Dict[str, Any]] = []
    update_interval = max(1, dataset_size // 100)
    for index in range(dataset_size):
        samples.append(builder.generate_sample())
        if (index + 1) % update_interval == 0 or index + 1 == dataset_size:
            progress = (index + 1) / dataset_size
            datasets.update_one({"_id": dataset_id}, {"$set": {"progress": progress}})

    payloads = [
        {"dataset": dataset_id, "data": sample, "created_at": datetime.utcnow()}
        for sample in samples
    ]

    for start in range(0, len(payloads), BULK_INSERT_SIZE):
        chunk = payloads[start : start + BULK_INSERT_SIZE]
        if chunk:
            data_collection.insert_many(chunk)

    datasets.update_one(
        {"_id": dataset_id},
        {"$set": {"status": "generated", "progress": 0.0}},
    )


def determine_dataset_size(size_info: Any, max_size: int, formats_count: int) -> int:
    formats_count = max(1, formats_count)
    if isinstance(size_info, Mapping):
        requested = size_info.get("size", max_size)
    else:
        requested = size_info or max_size

    if is_integer(requested):
        value = max(1, int(requested))
        return min(value, max_size)

    keyword = str(requested).lower()
    return calculate_size_from_keyword(keyword, max_size, formats_count)


def calculate_size_from_keyword(keyword: str, max_size: int, formats_size: int) -> int:
    match keyword:
        case "complete":
            return max_size
        case "advanced":
            return max_size // 2
        case "recommended":
            return max_size // formats_size
        case "small":
            return max_size // formats_size // 2
        case "tiny":
            return max(max_size // formats_size // 5, 1)
        case _:
            return min(max_size, 1000)


class DatasetBuilder:
    def __init__(
        self,
        *,
        configuration: Mapping[str, Any],
        db,
        entity_keys: Sequence[str],
        randomizers: Sequence[Mapping[str, Any]],
    ) -> None:
        self.configuration = configuration
        self.db = db
        self.entity_keys = list(entity_keys)
        self.randomizers = list(randomizers)

    def generate_sample(self) -> Dict[str, Any]:
        built_config = self._build_configuration(self.configuration)
        template = built_config["template"]
        attributes = built_config["attributes"]
        resolved_text, entities = self._render_entity(template, attributes)
        resolved_text = self._apply_randomizer(resolved_text)
        return {"text": resolved_text.strip(), "entities": entities}

    def _build_configuration(self, configuration: Mapping[str, Any]) -> Dict[str, Any]:
        template = random.choice(configuration.get("formats") or [""])
        attributes = configuration.get("attributes") or []
        built_attributes: List[Dict[str, Any]] = []

        for attribute in attributes:
            built_attr, extra_attrs = self._build_attribute(attribute)
            built_attributes.append(built_attr)
            built_attributes.extend(extra_attrs)

        return {"template": re.sub(r"\s+", " ", template.strip()), "attributes": built_attributes}

    def _build_attribute(self, attribute: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        key = attribute.get("key")
        frequency = float(attribute.get("frequency", 1))
        include = random.random() <= frequency
        requirements = attribute.get("requirements") or []

        value_spec = attribute.get("value") if include else None
        extra_attrs: List[Dict[str, Any]] = []
        value: Any = ""

        if isinstance(value_spec, Mapping):
            value, extra_attrs = self._build_dynamic_value(value_spec)
        elif value_spec is not None:
            value = value_spec

        requirement_ok = True
        if include and value not in (None, ""):
            requirement_ok = self._check_requirements(value, requirements)
            # if not requirement_ok:
            #     value = ""

        return (
            {"key": key, "value": "" if value is None else value, "requirements": requirement_ok},
            extra_attrs,
        )

    def _build_dynamic_value(self, spec: Mapping[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        value_type = spec.get("type", "string")
        rule = spec.get("rule")
        parameters = spec.get("parameters") or {}

        match rule:
            case "randint":
                minimum = int(parameters.get("min", 0))
                maximum = int(parameters.get("max", 100))
                if minimum > maximum:
                    minimum, maximum = maximum, minimum
                value = random.randint(minimum, maximum)
                return coerce_type(value_type, value), []
            case "alphanum":
                regex = parameters.get("regex", "")
                value = rstr.xeger(regex) if regex else ""
                return coerce_type(value_type, value), []
            case "data":
                data_id = parameters.get("object_id")
                if data_id:
                    record = self.db.get_collection("models_data").find_one({"_id": ObjectId(data_id)})
                    if record and record.get("data"):
                        value = random.choice(record["data"])
                        return coerce_type(value_type, value), []
                return "", []
            case "configuration":
                config_id = parameters.get("object_id")
                if config_id:
                    nested = self.db.get_collection("models_configurations").find_one({"_id": ObjectId(config_id)})
                    if nested:
                        built = self._build_configuration(nested)
                        return built.get("template", ""), built.get("attributes", [])
                return "", []
            case _:
                return "", []

    def _check_requirements(self, value: Any, requirements: Iterable[Mapping[str, Any]]) -> bool:
        for requirement in requirements or []:
            rule = requirement.get("rule")
            constraint = requirement.get("constraint")
            try:
                if rule == "regex":
                    if not re.match(str(constraint), str(value)):
                        return False
                elif rule == "eq" and str(value) != str(constraint):
                    return False
                elif rule == "neq" and str(value) == str(constraint):
                    return False
                elif rule == "gt" and float(value) <= float(constraint):
                    return False
                elif rule == "lt" and float(value) >= float(constraint):
                    return False
                elif rule == "gte" and float(value) < float(constraint):
                    return False
                elif rule == "lte" and float(value) > float(constraint):
                    return False
                elif rule == "in":
                    if str(value) not in split_constraint(constraint):
                        return False
                elif rule == "nin":
                    if str(value) in split_constraint(constraint):
                        return False
                elif rule == "contains" and str(constraint) not in str(value):
                    return False
                elif rule == "ncontains" and str(constraint) in str(value):
                    return False
            except Exception:
                return False
        return True

    def _render_entity(
        self,
        template: str,
        attributes: Sequence[Mapping[str, Any]],
    ) -> Tuple[str, List[List[Any]]]:
        attr_map = {attr.get("key"): attr for attr in attributes}
        resolved_values: Dict[str, str] = {}

        def resolve_value(key: str, stack: Optional[List[str]] = None) -> str:
            stack = stack or []
            if key in resolved_values:
                return resolved_values[key]
            if key in stack:
                return ""
            attr = attr_map.get(key)
            if not attr:
                resolved = ""
            else:
                raw_value = str(attr.get("value", ""))
                parts = []
                last = 0
                for match in PLACEHOLDER_PATTERN.finditer(raw_value):
                    parts.append(raw_value[last : match.start()])
                    nested_key = match.group("key")
                    parts.append(resolve_value(nested_key, stack + [key]))
                    last = match.end()
                parts.append(raw_value[last:])
                resolved = "".join(parts)
            resolved_values[key] = resolved
            return resolved

        parts: List[str] = []
        entities: List[List[Any]] = []
        cursor = 0
        last_index = 0

        for match in PLACEHOLDER_PATTERN.finditer(template):
            parts.append(template[last_index : match.start()])
            cursor += len(template[last_index : match.start()])

            key = match.group("key")
            value = resolve_value(key)
            attr = attr_map.get(key)
            requirements_met = True if attr is None else bool(attr.get("requirements", True))

            if key in self.entity_keys and value and requirements_met:
                start = cursor
                cursor += len(value)
                entities.append([start, cursor, key])
            else:
                cursor += len(value)

            parts.append(value)
            last_index = match.end()

        parts.append(template[last_index:])
        cursor += len(template[last_index:])

        final_text = "".join(parts)
        return final_text, entities

    def _apply_randomizer(self, text: str) -> str:
        if not self.randomizers:
            return text
        randomizer = random.choice(self.randomizers)
        frequency = float(randomizer.get("frequency", 1))
        if random.random() > frequency:
            return text
        rule = randomizer.get("rule")
        if rule == "upper":
            return text.upper()
        if rule == "lower":
            return text.lower()
        return text


def coerce_type(value_type: str, value: Any) -> Any:
    if value is None:
        return None
    if value_type == "number":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return str(value)


def split_constraint(constraint: Any) -> List[str]:
    if isinstance(constraint, str):
        return [part.strip() for part in constraint.split(",") if part.strip()]
    if isinstance(constraint, Iterable):
        return [str(item) for item in constraint]
    return [str(constraint)]


def is_integer(value: Any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def bump_version(version: str, bump: str) -> str:
    major, minor = map(int, version.split("."))
    if bump == "major":
        return f"{major + 1}.0"
    return f"{major}.{minor + 1}"
