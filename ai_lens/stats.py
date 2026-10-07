import math
import random
from statistics import mean
from typing import Any

MIN_RUNS = 3
RESAMPLES = 2000
EFFECT = 0.1


def _interval(samples: list[float]) -> tuple[float, float]:
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]


def _verdict(low: float, high: float, enough: bool) -> str:
    if not enough:
        return "not enough data"
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "no clear change"


def paired(deltas: list[float]) -> dict[str, Any]:
    rng = random.Random(0)
    samples = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(RESAMPLES)]
    low, high = _interval(samples)
    return {
        "delta": mean(deltas),
        "low": low,
        "high": high,
        "paired": True,
        "n": len(deltas),
        "verdict": _verdict(low, high, len(deltas) >= MIN_RUNS),
    }


def unpaired(before: list[float], after: list[float]) -> dict[str, Any]:
    rng = random.Random(0)
    samples = [
        mean(rng.choices(after, k=len(after))) - mean(rng.choices(before, k=len(before))) for _ in range(RESAMPLES)
    ]
    low, high = _interval(samples)
    return {
        "delta": mean(after) - mean(before),
        "low": low,
        "high": high,
        "paired": False,
        "n": min(len(before), len(after)),
        "verdict": _verdict(low, high, min(len(before), len(after)) >= MIN_RUNS),
    }


def compare(before: dict[str, float], after: dict[str, float], before_all: list[float], after_all: list[float]) -> dict[str, Any] | None:
    shared = [key for key in after if key in before]
    if len(shared) >= MIN_RUNS:
        return paired([after[key] - before[key] for key in shared])
    if before_all and after_all:
        return unpaired(before_all, after_all)
    return None


def inputs_needed(result: dict[str, Any] | None, effect: float = EFFECT) -> int | None:
    if result is None or result["n"] == 0:
        return None
    spread = (result["high"] - result["low"]) / 2
    target = max(MIN_RUNS, math.ceil(result["n"] * (spread / effect) ** 2))
    return max(0, target - result["n"])
