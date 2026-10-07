import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from . import llm, references
from .evaluators import Evaluator, for_kind
from .evaluators.geval import TOOL_USE, evaluation_steps
from .evaluators.reference import show_reference
from .store import load_runs, save_plan

Check = dict[str, Any]

GENERIC_CHECK = "Does the output fully and correctly deliver what the inputs asked for?"

DEFAULTS = {
    "text": ["not_empty", "geval"],
    "json": ["valid_json", "geval"],
    "image": ["image_integrity", "geval"],
    "video": ["video_integrity", "geval"],
    "audio": ["audio_integrity", "audio_silence", "geval"],
    "object": ["not_empty", "geval"],
}

PROMPT = """You are setting up automatic evaluation for an AI application.

What the developer wants to track:
{goal}

Output type: {kind}

Recent runs (inputs and output):
{examples}

Steps the application records inside a run:
{steps}

Evaluators you can use:
{evaluators}

Pick between 2 and 6 checks that together measure what the developer wants to track.
- Always include "success".
- Use "geval" for broad qualities only a careful reviewer can assess. Write each geval check as one criterion, such as "Coherence: the answer is well organised and each sentence follows from the last." Use one geval check per quality.
- Use "judge" for one narrow question about the output, such as "Does the video show a cat?".
- If the application calls tools, use "trajectory" to check the tool use, such as "Did it call the right tools with the right arguments, and use what they returned?".
- For evaluators that don't need a check, set check to an empty string."""

RUBRIC = """Study what makes these reference results good. Write 3 to 6 criteria a new output must meet to be as good as them. Make each criterion specific and checkable by looking at one output, like "The subject stays sharp and in frame for the whole clip" rather than "high quality". Give each a short name of 2 to 4 words."""

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["criteria"],
    "additionalProperties": False,
}

GENERIC_RUBRIC = [{"name": "Closeness", "description": "Matches the references in content, quality and style."}]


def _detect_kind(runs: list[dict[str, Any]]) -> str | None:
    kinds = Counter(run.get("output_type") for run in runs[-50:] if run.get("output_type") not in (None, "none"))
    return kinds.most_common(1)[0][0] if kinds else None


def _examples(runs: list[dict[str, Any]]) -> str:
    samples = [{"inputs": run.get("inputs"), "output": run.get("output")} for run in runs[-3:]]
    text = json.dumps(samples, default=str, indent=2)
    return text[:6_000] if samples else "No runs yet."


def _step_names(runs: list[dict[str, Any]]) -> dict[str, str]:
    return {step["name"]: step.get("type", "step") for run in runs[-50:] for step in run.get("steps") or []}


def _describe_steps(names: dict[str, str]) -> str:
    return ", ".join(f"{name} ({kind})" for name, kind in names.items()) or "None recorded."


def _default_checks(kind: str, tools: bool) -> list[Check]:
    checks = [{"evaluator": name, "check": GENERIC_CHECK if name == "geval" else ""} for name in DEFAULTS.get(kind, ["geval"])]
    return checks + [{"evaluator": "trajectory", "check": TOOL_USE}] if tools else checks


def _model_checks(goal: str, kind: str, runs: list[dict[str, Any]], options: list[Evaluator]) -> list[Check]:
    names = [option.name for option in options]
    schema = {
        "type": "object",
        "properties": {
            "checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"evaluator": {"type": "string", "enum": names}, "check": {"type": "string"}},
                    "required": ["evaluator", "check"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["checks"],
        "additionalProperties": False,
    }
    evaluators = "\n".join(f"- {option.name}: {option.description}" for option in options)
    prompt = PROMPT.format(goal=goal, kind=kind, examples=_examples(runs), steps=_describe_steps(_step_names(runs)), evaluators=evaluators)
    data, _ = llm.ask_json([prompt], schema)
    return data["checks"]


def _clean(checks: list[Check], options: list[Evaluator]) -> list[Check]:
    allowed = {option.name: option for option in options}
    cleaned = [{"evaluator": "success", "check": ""}]
    for check in checks:
        evaluator = allowed.get(check.get("evaluator", ""))
        if evaluator is None:
            continue
        text = check.get("check", "").strip() if evaluator.needs_check else ""
        item = {"evaluator": evaluator.name, "check": text}
        if item not in cleaned and (text or not evaluator.needs_check):
            cleaned.append(item)
    return [{"id": f"c{index}", **check} for index, check in enumerate(cleaned, start=1)]


def _rubric(goal: str, found: list[references.Reference]) -> list[Check]:
    parts: list[str | bytes] = [
        "You are writing a grading rubric for an AI application's outputs.",
        f"What the developer wants to track:\n{goal}",
    ]
    for index, reference in enumerate(found[:4], start=1):
        parts += show_reference(index, reference)
    parts.append(RUBRIC)
    try:
        data, _ = llm.ask_json(parts, RUBRIC_SCHEMA)
    except llm.LensError:
        return GENERIC_RUBRIC
    return data["criteria"][:6] or GENERIC_RUBRIC


def make_plan(goal: str, kind: str | None = None) -> dict[str, Any]:
    runs = load_runs()
    found = references.load()
    kind = kind or _detect_kind(runs) or "text"
    steps = _step_names(runs)
    options = [option for option in for_kind(kind) if option.name != "trajectory" or steps]
    try:
        checks, planner = _model_checks(goal, kind, runs, options), "model"
    except llm.LensError:
        checks, planner = _default_checks(kind, "tool" in steps.values()), "default"
    checks = _clean(checks, options)
    for check in checks:
        if check["evaluator"] in ("geval", "trajectory"):
            check["steps"] = evaluation_steps(goal, kind, check["evaluator"], check["check"])
    if found:
        checks.append({"id": f"c{len(checks) + 1}", "evaluator": "reference", "check": "", "rubric": _rubric(goal, found)})
    judge = llm.location()
    fingerprint = json.dumps({"goal": goal, "kind": kind, "checks": checks, "judge": judge}, sort_keys=True)
    plan = {
        "id": hashlib.sha1(fingerprint.encode()).hexdigest()[:10],
        "goal": goal,
        "kind": kind,
        "planner": planner,
        "judge": judge,
        "checks": checks,
        "references": references.fingerprint(found),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_plan(plan)
    return plan


def describe(check: Check) -> str:
    if check["evaluator"] == "reference":
        return f"reference: matches your best results ({len(check.get('rubric', []))} criteria)"
    return f"{check['evaluator']}: {check['check']}" if check.get("check") else check["evaluator"]
