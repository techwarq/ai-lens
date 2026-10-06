import os
import warnings
from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .usage import PRICES


@dataclass
class Settings:
    enabled: bool = True
    path: str | None = None
    copy_artifacts: bool = True
    model: Callable[[list[str | bytes]], Any] | None = None


settings = Settings()


def is_enabled() -> bool:
    return settings.enabled and os.environ.get("AI_LENS_DISABLED") != "1"


def configure(
    *,
    enabled: bool | None = None,
    path: str | None = None,
    copy_artifacts: bool | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
    references: str | Path | Iterable[dict[str, Any]] | None = None,
    model: Callable[[list[str | bytes]], Any] | str | None = None,
) -> None:
    if enabled is not None:
        settings.enabled = enabled
    if path is not None:
        settings.path = path
    if copy_artifacts is not None:
        settings.copy_artifacts = copy_artifacts
    if prices:
        PRICES.update(prices)
    if model is not None:
        from .llm import set_model

        set_model(model)
    if references is not None:
        from .references import sync

        try:
            sync(references)
        except (OSError, ValueError) as error:
            warnings.warn(f"ai-lens could not load references: {error}", RuntimeWarning, stacklevel=2)
