import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import media
from .store import data_dir, save_artifact

Reference = dict[str, Any]
Source = str | Path | Iterable[Reference]


def _items(source: Source) -> tuple[list[Reference], Path]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()], path.parent
    return list(source), Path.cwd()


def _normalize(item: Reference, base: Path) -> Reference:
    output = item.get("output")
    if output is None:
        raise ValueError("every reference needs an 'output'")
    if isinstance(output, str) and media.kind_of(output):
        path = media.as_media_path(base / output) or media.as_media_path(output)
        if path is None:
            raise ValueError(f"reference file not found: {output}")
        output = save_artifact(path, folder="references")
    return {
        "inputs": item.get("inputs") or {},
        "notes": item.get("notes", ""),
        "output": output,
        "kind": media.output_kind(output),
    }


def sync(source: Source) -> list[Reference]:
    items, base = _items(source)
    found = [_normalize(item, base) for item in items]
    text = json.dumps(found, indent=2, default=str)
    path = data_dir() / "references.json"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return found


def load() -> list[Reference]:
    path = data_dir() / "references.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(found: list[Reference]) -> str | None:
    if not found:
        return None
    return hashlib.sha1(json.dumps(found, sort_keys=True, default=str).encode()).hexdigest()[:10]


def matching(run: dict[str, Any], found: list[Reference], limit: int = 2) -> list[Reference]:
    inputs = run.get("inputs") or {}
    specific = [item for item in found if item["inputs"] and all(inputs.get(key) == value for key, value in item["inputs"].items())]
    general = [item for item in found if not item["inputs"]]
    return (specific or general)[:limit]
