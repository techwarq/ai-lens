import json
from typing import Any

from .. import llm
from .basic import score
from .judge import Parts, clip, show_inputs, show_output, unshowable

TOOL_USE = "Did the app call the right tools, in a sensible order, with the right arguments, and does its final output correctly use what the tools returned?"

RUBRIC = """Score from 0 to 10 using these bands:
- 0 to 2: fails the criterion.
- 3 to 6: partly meets it, with clear problems.
- 7 to 9: meets it, with minor gaps.
- 10: meets it completely."""

INSTRUCTION = """Follow the evaluation steps in order and write one short note per step about what you actually saw. Then give a whole-number score and a one-sentence reason that names what decided it."""

STEPS_PROMPT = """Write 3 to 5 evaluation steps a careful reviewer will follow to grade one output against this criterion. Each step is one sentence that says exactly what to look at and what counts as a problem. Only mention things the reviewer will be shown: {shown}."""

FALLBACK_STEPS = [
    "Read what the application was given and note what it was asked to do.",
    "Examine the output closely for everything the criterion asks about.",
    "List where it meets the criterion and where it falls short.",
    "Weigh how serious the shortfalls are.",
]

STEPS_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
    "required": ["steps"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["notes", "score", "reason"],
    "additionalProperties": False,
}

SHOWN = {
    "geval": "the inputs, the prompt, input images and the output (images directly, video as 4 frames, audio as a waveform and spectrogram)",
    "trajectory": "the inputs, every step the app took in order (tool calls with their arguments, results and errors) and the final output",
}


def evaluation_steps(goal: str, kind: str, evaluator: str, criterion: str) -> list[str]:
    parts: Parts = [
        "You are designing how a careful reviewer will grade outputs of an AI application.",
        f"What the developer wants to track:\n{goal}",
        f"Output type: {kind}",
        f"Criterion: {criterion}",
        STEPS_PROMPT.format(shown=SHOWN[evaluator]),
    ]
    try:
        data, _ = llm.ask_json(parts, STEPS_SCHEMA)
    except llm.LensError:
        return FALLBACK_STEPS
    found = [str(step).strip() for step in data["steps"] if str(step).strip()]
    return found[:5] or FALLBACK_STEPS


def _call(step: dict[str, Any]) -> str:
    arguments = ", ".join(f"{key}={json.dumps(value, default=str)}" for key, value in (step.get("inputs") or {}).items())
    line = f"[{step.get('type', 'step')}] {step.get('name')}({clip(arguments, 1_500)})"
    if step.get("status") == "error":
        error = step.get("error") or {}
        return f"{line} raised {error.get('type', 'Error')}: {clip(str(error.get('message', '')), 500)}"
    return f"{line} returned {clip(json.dumps(step.get('output'), default=str), 2_000)}"


def show_steps(run: dict[str, Any]) -> str:
    steps = run.get("steps") or []
    depth: dict[str, int] = {}
    lines = []
    for number, step in enumerate(steps, start=1):
        level = depth.get(step.get("parent_id") or "", -1) + 1
        depth[step["id"]] = level
        lines.append(f"{'  ' * level}{number}. {_call(step)}")
    return clip("\n".join(lines), 30_000)


def _grade(run: dict[str, Any], check: dict[str, Any], shown: Parts) -> dict[str, Any]:
    steps = "\n".join(f"{number}. {step}" for number, step in enumerate(check.get("steps") or FALLBACK_STEPS, start=1))
    parts: Parts = [
        "You are grading one run of an AI application.",
        f"Criterion: {check['check']}",
        f"Evaluation steps:\n{steps}",
        RUBRIC,
        *shown,
        INSTRUCTION,
    ]
    data, cost = llm.ask_json(parts, SCHEMA)
    rating = min(10, max(0, round(float(data["score"]))))
    return {**score(rating / 10, str(data["reason"]), cost), "notes": [str(note) for note in data["notes"]]}


def geval(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    kind = run.get("output_type")
    output = show_output(run.get("output"), kind)
    if output is None:
        return unshowable(kind)
    return _grade(run, check, ["What the application was given:", *show_inputs(run), "What it produced:", *output])


def trajectory(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    if not run.get("steps"):
        return score(None, "no steps recorded; mark tool calls with @lens.step(type=\"tool\")")
    kind = run.get("output_type")
    output = show_output(run.get("output"), kind) or [f"A {kind} output that can't be shown here."]
    shown: Parts = [
        "What the application was given:",
        *show_inputs(run),
        f"Every step it took, in order (nested steps are indented):\n{show_steps(run)}",
        "Its final output:",
        *output,
    ]
    return _grade(run, check, shown)
