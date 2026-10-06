from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import audio, basic, image, judge, reference, text, video

ALL = frozenset({"text", "json", "image", "video", "audio", "object", "none"})


@dataclass(frozen=True)
class Evaluator:
    name: str
    kinds: frozenset[str]
    description: str
    run: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    needs_check: bool = False
    plannable: bool = True


EVALUATORS = {
    evaluator.name: evaluator
    for evaluator in (
        Evaluator("success", ALL, "The run finished without raising an error.", basic.success),
        Evaluator(
            "judge",
            frozenset({"text", "json", "image", "video", "object"}),
            "A model grades the output against one written check, 0 to 1. Sees images and sampled video frames.",
            judge.judge,
            needs_check=True,
        ),
        Evaluator(
            "reference",
            frozenset({"text", "json", "image", "video", "object"}),
            "A model compares the output with your reference results using a rubric learned from them.",
            reference.compare,
            plannable=False,
        ),
        Evaluator("not_empty", frozenset({"text", "json", "object"}), "The output is not empty.", text.not_empty),
        Evaluator("valid_json", frozenset({"text", "json"}), "The output is valid JSON.", text.valid_json),
        Evaluator("image_integrity", frozenset({"image"}), "The image decodes and has a sane size.", image.integrity),
        Evaluator("video_integrity", frozenset({"video"}), "The video decodes, has frames and a non-zero duration.", video.integrity),
        Evaluator("audio_integrity", frozenset({"audio"}), "The audio decodes and has a non-zero duration.", audio.integrity),
        Evaluator("audio_silence", frozenset({"audio"}), "Share of the clip that is not silence.", audio.silence),
    )
}


def for_kind(kind: str) -> list[Evaluator]:
    return [evaluator for evaluator in EVALUATORS.values() if kind in evaluator.kinds and evaluator.plannable]
