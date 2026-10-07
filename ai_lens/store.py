import hashlib
import json
import mimetypes
import os
import shutil
import threading
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .models import Run

_lock = threading.Lock()


def data_dir() -> Path:
    path = Path(settings.path or os.environ.get("AI_LENS_DIR", ".ai-lens"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, (set, frozenset)):
        return list(value)
    if is_dataclass(value) and not isinstance(value, type):
        return vars(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return repr(value)[:2000]


def _append(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    text = "".join(json.dumps(row, default=_default) + "\n" for row in rows)
    with _lock, open(data_dir() / name, "a", encoding="utf-8") as file:
        file.write(text)


def _read(name: str) -> list[dict[str, Any]]:
    path = data_dir() / name
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_run(run: Run) -> None:
    _append("runs.jsonl", [{**vars(run), "steps": [vars(step) for step in run.steps]}])


def load_runs() -> list[dict[str, Any]]:
    return _read("runs.jsonl")


def save_evals(rows: list[dict[str, Any]]) -> None:
    _append("evals.jsonl", rows)


def load_evals() -> list[dict[str, Any]]:
    return _read("evals.jsonl")


def save_pairs(rows: list[dict[str, Any]]) -> None:
    _append("pairwise.jsonl", rows)


def load_pairs() -> list[dict[str, Any]]:
    return _read("pairwise.jsonl")


def save_summaries(rows: list[dict[str, Any]]) -> None:
    _append("summaries.jsonl", rows)


def load_summaries() -> list[dict[str, Any]]:
    return _read("summaries.jsonl")


def save_plan(plan: dict[str, Any]) -> None:
    (data_dir() / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def load_plan() -> dict[str, Any] | None:
    path = data_dir() / "plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_artifact(path: Path, name: str | None = None, folder: str = "artifacts") -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "mime": mimetypes.guess_type(path.name)[0],
    }
    if settings.copy_artifacts:
        target = data_dir() / folder / f"{name or info['sha256'][:16]}{path.suffix.lower()}"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        info["artifact"] = str(target)
    return info
