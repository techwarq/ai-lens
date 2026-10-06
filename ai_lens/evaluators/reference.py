import json
from typing import Any

from .. import llm, references
from .basic import clamp, score
from .judge import Parts, clip, show_inputs, show_output, unshowable

SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "score": {"type": "number"}, "reason": {"type": "string"}},
                "required": ["name", "score", "reason"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["criteria", "score", "reason"],
    "additionalProperties": False,
}

INSTRUCTION = """Score the new output on each rubric criterion from 0 to 1, where 1 means it meets that criterion as well as the references do. Use the criterion names exactly as written. Then give an overall score from 0 to 1 for how close the new output comes to the reference standard, and a one-sentence reason that names the biggest gap."""


def show_reference(index: int, reference: dict[str, Any]) -> Parts:
    notes = f" (developer's notes: {reference['notes']})" if reference.get("notes") else ""
    parts: Parts = [f"Reference {index}{notes}:"]
    if reference.get("inputs"):
        parts.append("Made from: " + clip(json.dumps(reference["inputs"], default=str), 2_000))
    return parts + (show_output(reference["output"], reference["kind"]) or [clip(str(reference["output"]), 4_000)])


def compare(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    matches = references.matching(run, references.load())
    if not matches:
        return score(None, "no reference matches these inputs")
    kind = run.get("output_type")
    output = show_output(run.get("output"), kind)
    if output is None:
        return unshowable(kind)
    rubric = "\n".join(f"- {item['name']}: {item['description']}" for item in check.get("rubric", []))
    parts: Parts = [
        "You are grading a new output of an AI application against reference outputs the developer considers their best results.",
        f"Rubric:\n{rubric}",
    ]
    for index, reference in enumerate(matches, start=1):
        parts += show_reference(index, reference)
    parts += ["The new output was made from:", *show_inputs(run), "The new output:", *output, INSTRUCTION]
    data, cost = llm.ask_json(parts, SCHEMA)
    criteria = {str(item["name"]): clamp(item["score"]) for item in data["criteria"]}
    return {**score(clamp(data["score"]), str(data["reason"]), cost), "criteria": criteria}
