from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ki_geodaten.models import VectorOptions


def dumps_or_none(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def loads_json(raw: str | None) -> Any | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def loads_json_object(raw: str | None) -> dict | None:
    payload = loads_json(raw)
    return payload if isinstance(payload, dict) else None


def row_value(row: Mapping | Any, key: str) -> Any | None:
    keys = row.keys() if hasattr(row, "keys") else ()
    return row[key] if key in keys else None


def vector_options_payload(row: Mapping | Any) -> dict | None:
    payload = loads_json_object(row_value(row, "vector_options"))
    if payload is not None:
        return payload

    metadata = loads_json_object(row_value(row, "run_metadata"))
    if metadata is None:
        return None
    vector_options = metadata.get("vector_options")
    return vector_options if isinstance(vector_options, dict) else None


def vector_options_model(row: Mapping | Any) -> VectorOptions:
    return VectorOptions.model_validate(vector_options_payload(row) or {})
