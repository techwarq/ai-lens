import importlib
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import settings
from .store import data_dir
from .usage import extract, price

Parts = list[str | bytes]
Model = Callable[[Parts], Any]

JSON_INSTRUCTION = "Reply with only a JSON object and no other text. It must match this JSON schema:\n{schema}"
RETRY = "Your last reply was not a valid JSON object for that schema. Reply again with only the JSON object."
NO_MODEL = (
    "No model configured. Give ai-lens your own model function with lens.configure(model=...) "
    "or `lens track --model your_module:your_function`."
)


class LensError(Exception):
    pass


def _config_path() -> Path:
    return data_dir() / "config.json"


def _read_config() -> dict[str, Any]:
    path = _config_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _location(model: Model) -> str | None:
    module = getattr(model, "__module__", None)
    name = getattr(model, "__qualname__", None)
    if not module or not name or module == "__main__" or "<locals>" in name:
        return None
    return f"{module}:{name}"


def set_model(model: Model | str) -> None:
    location = model if isinstance(model, str) else _location(model)
    settings.model = None if isinstance(model, str) else model
    if location is None:
        warnings.warn(
            "ai-lens: put your model function at the top level of an importable module, "
            "not in the script you run, so the `lens` command can use it too.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    _config_path().write_text(json.dumps({**_read_config(), "model": location}, indent=2), encoding="utf-8")


def _import(location: str) -> Model:
    module_name, _, attribute = location.partition(":")
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def location() -> str | None:
    if settings.model is not None:
        return _location(settings.model) or getattr(settings.model, "__qualname__", "in-process function")
    return _read_config().get("model")


def load_model() -> Model:
    if settings.model is not None:
        return settings.model
    location = _read_config().get("model")
    if not location:
        raise LensError(NO_MODEL)
    try:
        return _import(location)
    except (ImportError, AttributeError) as error:
        raise LensError(f"Could not load your model function '{location}': {error}") from error


def _text(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    content = getattr(reply, "content", None)
    if isinstance(content, list):
        return "".join(getattr(block, "text", "") for block in content)
    choices = getattr(reply, "choices", None)
    if choices:
        return choices[0].message.content or ""
    text = getattr(reply, "text", None)
    if isinstance(text, str):
        return text
    raise LensError("Your model function must return the model's reply as text.")


def _parse(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict) or any(key not in data for key in schema.get("required", [])):
        raise ValueError("reply is missing required fields")
    return data


def ask_json(parts: Parts, schema: dict[str, Any]) -> tuple[dict[str, Any], float]:
    model = load_model()
    request: Parts = [*parts, JSON_INSTRUCTION.format(schema=json.dumps(schema))]
    cost = 0.0
    for _ in range(2):
        try:
            reply = model(request)
        except Exception as error:
            raise LensError(f"Your model raised {type(error).__name__}: {error}") from error
        found = extract(reply)
        cost += (price(found) if found else None) or 0.0
        try:
            return _parse(_text(reply), schema), cost
        except ValueError:
            request = [*request, RETRY]
    raise LensError("Your model did not reply with valid JSON.")
