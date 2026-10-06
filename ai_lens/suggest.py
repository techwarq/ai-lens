import json
from pathlib import Path
from typing import Any

from . import git, llm
from .report import Row

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "impact": {"type": "string", "enum": ["quality", "cost", "latency", "reliability"]},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "problem": {"type": "string"},
                    "change": {"type": "string"},
                    "expected": {"type": "string"},
                },
                "required": ["title", "impact", "file", "line", "problem", "change", "expected"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "suggestions"],
    "additionalProperties": False,
}

PROMPT = """You are a senior engineer helping a developer improve an AI application. They traced every run, scored the outputs, and now want to know what to change.

What they track:
{goal}

Report. Versions are in time order, scores run from 0 to 1, costs are dollars per run:
{report}

Recent prompts sent to the generator, newest last:
{prompts}

{diff}

Code of the traced functions and steps, with line numbers:
{code}

Suggest concrete changes, most valuable first.
- For each quality regression, name the change that most likely caused it and how to fix it.
- For token and cost waste, name the exact call and how to cut it: prompt caching for repeated context, trimming what is sent, removing repeated identical calls, lower max_tokens or effort, or a smaller model where the scores show quality would hold.
- For failures, name the cause and the fix.
- Every suggestion must point at a file and line from the code above and give the change as a short code snippet or a precise instruction.
- Only suggest what the evidence supports. If something is fine, leave it alone."""


def _code(runs: list[Row], limit: int = 8, lines: int = 50) -> str:
    sources: list[str] = []
    for run in runs[-20:]:
        for source in [run.get("source"), *(step.get("source") for step in run.get("steps") or [])]:
            if source and source not in sources:
                sources.append(source)
    blocks = []
    for source in sources[:limit]:
        file, _, line = source.rpartition(":")
        path = Path(file)
        if not path.is_file() or not line.isdigit():
            continue
        start = int(line)
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()[start - 1 : start - 1 + lines]
        numbered = "\n".join(f"{start + index:>5}  {content}" for index, content in enumerate(text))
        blocks.append(f"### {source}\n{numbered}")
    return "\n\n".join(blocks) or "No source available."


def _regression_diff(report: Row) -> str:
    worse = [change for change in report["changes"] if change["stats"] and change["stats"]["verdict"] == "worse"]
    if not worse:
        return ""
    change = min(worse, key=lambda change: change["stats"]["delta"])
    versions = {version["label"]: version for version in report["versions"]}
    before, after = versions[change["from"]], versions[change["to"]]
    if not before["commit"]:
        return ""
    target = None if after["dirty"] else after["commit"]
    text = git.diff(before["commit"], target)
    if not text:
        return ""
    return f"Code diff for the biggest regression ({change['from']} → {change['to']}):\n{text}"


def _prompts(runs: list[Row], limit: int = 5) -> str:
    prompts = list(dict.fromkeys(run["prompt"] for run in runs if run.get("prompt")))[-limit:]
    return "\n---\n".join(prompt[:2_000] for prompt in prompts) or "None recorded."


def _compact(report: Row) -> str:
    keep = {key: report[key] for key in ("checks", "versions", "changes", "overall", "hotspots", "failures")}
    return json.dumps(keep, default=str, indent=1)


def suggest(report: Row, runs: list[Row]) -> tuple[Row, float]:
    prompt = PROMPT.format(
        goal=report["goal"],
        report=_compact(report),
        prompts=_prompts(runs),
        diff=_regression_diff(report),
        code=_code(runs),
    )
    return llm.ask_json([prompt], SCHEMA)


def render(result: Row) -> str:
    lines = [result["summary"].strip()]
    for index, item in enumerate(result["suggestions"], start=1):
        lines += [
            "",
            f"{index}. {item['title']}  [{item['impact']}]",
            f"   {item['file']}:{item['line']}",
            f"   Problem:  {item['problem']}",
            f"   Change:   {item['change']}",
            f"   Expected: {item['expected']}",
        ]
    return "\n".join(lines)
