from __future__ import annotations

import functools
import inspect
import sys
import time
import traceback
import uuid
import warnings
from collections.abc import Callable, Sequence
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar, cast, overload

from . import media, store, usage
from .config import is_enabled
from .git import git_info
from .models import Run, Step

F = TypeVar("F", bound=Callable[..., Any])

LOGGABLE = {"output", "model", "input_tokens", "output_tokens", "cost", "inputs"}

_current: ContextVar[_Span | None] = ContextVar("ai_lens_span", default=None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _location(filename: str, line: int) -> str:
    path = Path(filename)
    try:
        path = path.resolve().relative_to(Path.cwd())
    except ValueError:
        pass
    return f"{path}:{line}"


def _source(fn: Callable[..., Any]) -> str | None:
    code = getattr(inspect.unwrap(fn), "__code__", None)
    if code is None:
        return None
    return _location(code.co_filename, code.co_firstlineno)


def _signature(fn: Callable[..., Any]) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _inputs(signature: inspect.Signature | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if signature is None:
        return {"args": list(args), "kwargs": kwargs}
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    bound.apply_defaults()
    return {key: value for key, value in bound.arguments.items() if key not in ("self", "cls")}


class _Span:
    def __init__(
        self,
        kind: str,
        name: str,
        type: str,
        tags: Sequence[str],
        source: str | None,
        inputs: dict[str, Any] | None,
    ) -> None:
        self.kind = kind
        self.name = name
        self.type = type
        self.tags = tags
        self.source = source
        self.inputs = inputs
        self.run: Run | None = None
        self.record: Run | Step | None = None
        self.counted: set[int] = set()
        self._token: Token[_Span | None] | None = None
        self._start = 0.0

    def __enter__(self) -> _Span:
        parent = _current.get()
        if not is_enabled() or (parent is None and self.kind == "step"):
            return self
        if parent is None or parent.run is None:
            self.run = Run(
                id=uuid.uuid4().hex,
                name=self.name,
                source=self.source,
                started_at=_now(),
                inputs=self.inputs,
                tags=list(self.tags),
            )
            self.record = self.run
        else:
            self.run = parent.run
            self.counted = parent.counted
            self.record = Step(
                id=uuid.uuid4().hex,
                name=self.name,
                parent_id=parent.record.id if isinstance(parent.record, Step) else None,
                type=self.type,
                source=self.source,
                started_at=_now(),
                inputs=self.inputs,
            )
            self.run.steps.append(self.record)
        self._token = _current.set(self)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        record = self.record
        if record is None or self._token is None:
            return False
        _current.reset(self._token)
        record.latency_s = time.perf_counter() - self._start
        if exc_type is not None and exc is not None:
            record.status = "error"
            record.error = {
                "type": exc_type.__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            }
        if isinstance(record, Run):
            _finish(record)
        return False

    def output(self, value: Any) -> Any:
        if self.record is not None:
            self.record.output = value
            self._count(value)
        return value

    def log(self, **fields: Any) -> None:
        record, run = self.record, self.run
        if record is None or run is None:
            return
        for key, value in fields.items():
            if key == "usage":
                self._count(value)
            elif key == "params":
                run.params.update(value)
            elif key == "prompt":
                run.prompt = value
            elif key in ("tags", "assets"):
                getattr(run, key).extend([value] if isinstance(value, (str, Path)) else value)
            elif key == "metadata":
                record.metadata.update(value)
            elif key in LOGGABLE:
                setattr(record, key, value)
            else:
                record.metadata[key] = value

    def _count(self, value: Any) -> None:
        found = usage.extract(value)
        record = self.record
        if found is None or record is None or id(value) in self.counted:
            return
        self.counted.add(id(value))
        record.model = record.model or found.model
        record.input_tokens = (record.input_tokens or 0) + found.total_input_tokens
        record.output_tokens = (record.output_tokens or 0) + found.output_tokens
        cost = usage.price(found)
        if cost is not None:
            record.cost = (record.cost or 0.0) + cost
        if isinstance(record, Step) and record.type == "step":
            record.type = "llm"


def _sum(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _input_files(inputs: dict[str, Any]) -> list[tuple[str, Path]]:
    found = []
    for name, value in inputs.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        found += [(name, path) for item in values if (path := media.as_media_path(item)) is not None]
    return found


def _collect_inputs(run: Run) -> None:
    inputs = run.inputs or {}
    if run.prompt is None and isinstance(inputs.get("prompt"), str):
        run.prompt = inputs["prompt"]
    logged = [("logged", path) for item in run.assets if (path := media.as_media_path(item)) is not None]
    files: dict[Path, str] = {}
    for name, path in _input_files(inputs) + logged:
        files.setdefault(path.resolve(), name)
    run.assets = [{"input": name, **store.save_artifact(path, folder="assets")} for path, name in files.items()]


def _finish(run: Run) -> None:
    try:
        run.ended_at = _now()
        run.git_commit, run.git_branch, run.git_dirty, run.git_edits, run.git_edited = git_info()
        records: list[Run | Step] = [run, *run.steps]
        run.input_tokens = _sum([record.input_tokens for record in records])
        run.output_tokens = _sum([record.output_tokens for record in records])
        run.cost = _sum([record.cost for record in records])
        run.model = run.model or next((step.model for step in run.steps if step.model), None)
        path = media.as_media_path(run.output)
        if path is not None:
            run.output = store.save_artifact(path, run.id)
        run.output_type = media.output_kind(run.output)
        _collect_inputs(run)
        store.save_run(run)
    except Exception as error:
        warnings.warn(f"ai-lens could not save run '{run.name}': {error}", RuntimeWarning, stacklevel=3)


class _Spec:
    def __init__(self, kind: str, name: str | None, type: str, tags: Sequence[str]) -> None:
        self.kind = kind
        self.name = name
        self.type = type
        self.tags = tags
        self._span: _Span | None = None

    def __call__(self, fn: F) -> F:
        return _wrap(fn, self)

    def __enter__(self) -> _Span:
        caller = sys._getframe(1)
        source = _location(caller.f_code.co_filename, caller.f_lineno)
        self._span = _Span(self.kind, self.name or self.kind, self.type, self.tags, source, None)
        return self._span.__enter__()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        span, self._span = self._span, None
        return span.__exit__(exc_type, exc, tb) if span else False


def _active(kind: str) -> bool:
    return is_enabled() and (kind == "trace" or _current.get() is not None)


def _wrap(fn: F, spec: _Spec) -> F:
    name = spec.name or fn.__name__
    source = _source(fn)
    signature = _signature(fn)

    def open_span(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Span:
        return _Span(spec.kind, name, spec.type, spec.tags, source, _inputs(signature, args, kwargs))

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _active(spec.kind):
                return await fn(*args, **kwargs)
            with open_span(args, kwargs) as span:
                return span.output(await fn(*args, **kwargs))

        return cast(F, async_wrapper)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _active(spec.kind):
            return fn(*args, **kwargs)
        with open_span(args, kwargs) as span:
            return span.output(fn(*args, **kwargs))

    return cast(F, wrapper)


@overload
def trace(target: F, /) -> F: ...
@overload
def trace(target: str | None = None, /, *, name: str | None = None, tags: Sequence[str] = ()) -> _Spec: ...
def trace(target: Any = None, /, *, name: str | None = None, tags: Sequence[str] = ()) -> Any:
    if callable(target):
        return _wrap(target, _Spec("trace", name, "step", tags))
    return _Spec("trace", target or name, "step", tags)


@overload
def step(target: F, /) -> F: ...
@overload
def step(target: str | None = None, /, *, name: str | None = None, type: str = "step") -> _Spec: ...
def step(target: Any = None, /, *, name: str | None = None, type: str = "step") -> Any:
    if callable(target):
        return _wrap(target, _Spec("step", name, type, ()))
    return _Spec("step", target or name, type, ())


def log(**fields: Any) -> None:
    span = _current.get()
    if span is not None:
        span.log(**fields)
