from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from .evaluators import EVALUATORS
from .evaluators.basic import score
from .llm import LensError
from .report import version_key
from .store import load_evals, save_evals


def pending(
    plan: dict[str, Any],
    runs: list[dict[str, Any]],
    evals: list[dict[str, Any]],
    per_version: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    done = {(row["run_id"], row["check_id"]) for row in evals if row.get("plan_id") == plan["id"]}
    scored = {row["run_id"] for row in evals if row.get("plan_id") == plan["id"]}
    by_version: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for run in sorted(runs, key=lambda run: run.get("started_at") or "", reverse=True):
        by_version[version_key(run)].append(run)
    jobs = []
    for members in by_version.values():
        budget = max(0, per_version - sum(run["id"] in scored for run in members))
        chosen = [run for run in members if run["id"] in scored] + [run for run in members if run["id"] not in scored][:budget]
        jobs += [(run, check) for run in chosen for check in plan["checks"] if (run["id"], check["id"]) not in done]
    return jobs


def _score(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    evaluator = EVALUATORS.get(check["evaluator"])
    if evaluator is None:
        raise LensError(f"Unknown evaluator '{check['evaluator']}' in plan.json")
    if evaluator.name != "success" and run.get("status") != "ok":
        return score(None, "skipped because the run failed")
    return evaluator.run(run, check)


def evaluate(
    plan: dict[str, Any],
    runs: list[dict[str, Any]],
    per_version: int = 20,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], list[str]]:
    jobs = pending(plan, runs, load_evals(), per_version)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score, run, check): (run, check) for run, check in jobs}
        for future in as_completed(futures):
            run, check = futures[future]
            try:
                result = future.result()
            except LensError as error:
                errors.append(str(error))
                continue
            except Exception as error:
                errors.append(f"{check['evaluator']} failed: {type(error).__name__}: {error}")
                continue
            rows.append(
                {
                    "plan_id": plan["id"],
                    "run_id": run["id"],
                    "check_id": check["id"],
                    **result,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
    save_evals(rows)
    return rows, errors
