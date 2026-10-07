from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import audio, basic, geval, image, judge, reference, text, video

ALL = frozenset({"text", "json", "image", "video", "audio", "object", "none"})
OUTPUTS = ALL - {"none"}


@dataclass(frozen=True)
class Evaluator:
    name: str
    kinds: frozenset[str]
    description: str
    run: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    needs_check: bool = False
    plannable: bool = True
    gate: bool = False


EVALUATORS = {
    evaluator.name: evaluator
    for evaluator in (
        Evaluator("success", ALL, "The run finished without raising an error.", basic.success, gate=True),
        Evaluator(
            "judge",
            OUTPUTS,
            "A model answers one narrow question about the output, 0 to 1.",
            judge.judge,
            needs_check=True,
        ),
        Evaluator(
            "geval",
            OUTPUTS,
            "G-Eval: a model follows evaluation steps written for one criterion and scores the output 0 to 10. Best for broad qualities like coherence, faithfulness to the prompt or visual quality.",
            geval.geval,
            needs_check=True,
        ),
        Evaluator(
            "trajectory",
            ALL,
            "G-Eval over the whole run: every step and tool call with its arguments and results. Use for how the app reached its output, like whether tool use was correct.",
            geval.trajectory,
            needs_check=True,
        ),
        Evaluator(
            "reference",
            OUTPUTS,
            "A model compares the output with your reference results using a rubric learned from them.",
            reference.compare,
            plannable=False,
        ),
        Evaluator("not_empty", frozenset({"text", "json", "object"}), "The output is not empty.", text.not_empty, gate=True),
        Evaluator("valid_json", frozenset({"text", "json"}), "The output is valid JSON.", text.valid_json, gate=True),
        Evaluator("image_integrity", frozenset({"image"}), "The image decodes and has a sane size.", image.integrity, gate=True),
        Evaluator("video_integrity", frozenset({"video"}), "The video decodes, has frames and a non-zero duration.", video.integrity, gate=True),
        Evaluator("audio_integrity", frozenset({"audio"}), "The audio decodes and has a non-zero duration.", audio.integrity, gate=True),
        Evaluator("audio_silence", frozenset({"audio"}), "Share of the clip that is not silence.", audio.silence),
    )
}


def for_kind(kind: str) -> list[Evaluator]:
    return [evaluator for evaluator in EVALUATORS.values() if kind in evaluator.kinds and evaluator.plannable]
