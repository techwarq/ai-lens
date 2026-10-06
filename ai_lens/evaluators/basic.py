from typing import Any


def score(value: float | None, reason: str, cost: float = 0.0) -> dict[str, Any]:
    return {"score": value, "reason": reason, "cost": cost}


def clamp(value: Any) -> float:
    return min(1.0, max(0.0, float(value)))


def success(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    if run.get("status") == "ok":
        return score(1.0, "completed")
    error = run.get("error") or {}
    return score(0.0, f"{error.get('type', 'Error')}: {error.get('message', '')}"[:300])
