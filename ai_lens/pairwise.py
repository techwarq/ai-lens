from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import pairwise

from . import llm
from .evaluators.judge import Parts, show_inputs, show_output
from .plan import describe
from .report import Row, groups, inputs_key
from .store import load_pairs, save_pairs

SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}, "reason": {"type": "string"}},
    "required": ["winner", "reason"],
    "additionalProperties": False,
}

INSTRUCTION = """Which output is better for what the developer cares about? Judge the outputs only, not their order or length. Answer "A", "B" or "tie", with a one-sentence reason that names the concrete difference."""


def _ask(plan: Row, before: Row, after: Row, swap: bool) -> tuple[str, str, float]:
    first, second = (after, before) if swap else (before, after)
    checks = "\n".join(f"- {describe(check)}" for check in plan["checks"] if check["evaluator"] != "success")
    parts: Parts = [
        "You are comparing two outputs an AI application made from the same inputs.",
        f"What the developer cares about:\n{plan['goal']}\n{checks}",
        "Both outputs were made from:",
        *show_inputs(after),
        "Output A:",
        *(show_output(first.get("output"), first.get("output_type")) or []),
        "Output B:",
        *(show_output(second.get("output"), second.get("output_type")) or []),
        INSTRUCTION,
    ]
    data, cost = llm.ask_json(parts, SCHEMA)
    winner = {"A": first, "B": second}.get(data["winner"])
    side = "tie" if winner is None else ("after" if winner is after else "before")
    return side, str(data["reason"]), cost


def judge_pair(plan: Row, before: Row, after: Row) -> Row:
    one, two = _ask(plan, before, after, swap=False), _ask(plan, before, after, swap=True)
    consistent = one[0] == two[0]
    return {
        "plan_id": plan["id"],
        "before_run": before["id"],
        "after_run": after["id"],
        "winner": one[0] if consistent else "tie",
        "consistent": consistent,
        "reason": one[1] if consistent else f"changed its answer when the order was swapped: {one[1]}",
        "cost": one[2] + two[2],
        "at": datetime.now(timezone.utc).isoformat(),
    }


def _comparable(run: Row) -> bool:
    return run.get("status") == "ok" and run.get("output_type") not in ("audio", "none", None)


def candidates(plan: Row, runs: list[Row], per_change: int) -> list[tuple[Row, Row]]:
    done = {(row["before_run"], row["after_run"]) for row in load_pairs() if row.get("plan_id") == plan["id"]}
    named = groups(runs)
    jobs: list[tuple[Row, Row]] = []
    for old, new in pairwise(named.values()):
        latest = {inputs_key(run): run for run in old if _comparable(run)}
        matched: dict[str, tuple[Row, Row]] = {}
        for run in reversed(new):
            key = inputs_key(run)
            if _comparable(run) and key in latest and key not in matched:
                matched[key] = (latest[key], run)
        pairs = list(matched.values())
        finished = sum((before["id"], after["id"]) in done for before, after in pairs)
        todo = [pair for pair in pairs if (pair[0]["id"], pair[1]["id"]) not in done]
        jobs += todo[: max(0, per_change - finished)]
    return jobs


def run(plan: Row, runs: list[Row], per_change: int = 5, workers: int = 4) -> tuple[list[Row], list[str]]:
    rows: list[Row] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(judge_pair, plan, before, after) for before, after in candidates(plan, runs, per_change)]
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except llm.LensError as error:
                errors.append(str(error))
            except Exception as error:
                errors.append(f"side-by-side failed: {type(error).__name__}: {error}")
    save_pairs(rows)
    return rows, errors

