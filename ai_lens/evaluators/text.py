import json
from typing import Any

from .basic import score


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def not_empty(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    text = as_text(run.get("output")).strip()
    return score(1.0 if text else 0.0, f"{len(text)} characters")


def valid_json(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    output = run.get("output")
    if isinstance(output, (dict, list)):
        return score(1.0, "structured output")
    try:
        json.loads(as_text(output))
    except json.JSONDecodeError as error:
        return score(0.0, f"invalid JSON: {error.msg}")
    return score(1.0, "parses as JSON")
