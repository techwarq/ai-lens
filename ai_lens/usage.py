from typing import Any, NamedTuple

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

CACHE_WRITE_RATE = 1.25
CACHE_READ_RATE = 0.1


class Usage(NamedTuple):
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


def _get(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _first(source: Any, *keys: str) -> int | None:
    for key in keys:
        value = _get(source, key)
        if isinstance(value, int):
            return value
    return None


def extract(value: Any) -> Usage | None:
    usage = _get(value, "usage")
    if usage is None:
        return None
    input_tokens = _first(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first(usage, "output_tokens", "completion_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    model = _get(value, "model")
    return Usage(
        model=model if isinstance(model, str) else None,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        cache_write_tokens=_first(usage, "cache_creation_input_tokens") or 0,
        cache_read_tokens=_first(usage, "cache_read_input_tokens") or 0,
    )


def price(usage: Usage) -> float | None:
    if usage.model is None:
        return None
    key = next((name for name in sorted(PRICES, key=len, reverse=True) if name in usage.model), None)
    if key is None:
        return None
    input_rate, output_rate = PRICES[key]
    dollars = (
        usage.input_tokens * input_rate
        + usage.cache_write_tokens * input_rate * CACHE_WRITE_RATE
        + usage.cache_read_tokens * input_rate * CACHE_READ_RATE
        + usage.output_tokens * output_rate
    )
    return dollars / 1_000_000
