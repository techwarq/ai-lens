from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    id: str
    name: str
    parent_id: str | None = None
    type: str = "step"
    source: str | None = None
    started_at: str | None = None
    latency_s: float | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None
    inputs: dict[str, Any] | None = None
    output: Any = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    id: str
    name: str
    v: int = 1
    source: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    latency_s: float | None = None
    inputs: dict[str, Any] | None = None
    prompt: str | None = None
    assets: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    output: Any = None
    output_type: str | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
