import json
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise
from statistics import mean
from typing import Any

from . import git, stats
from .plan import describe

THRESHOLD = 0.05

Row = dict[str, Any]
Scores = dict[tuple[str, str], Row]


def version_key(run: Row) -> tuple[str, bool, str, str]:
    params = json.dumps(run.get("params") or {}, sort_keys=True, default=str)
    return run.get("git_commit") or "", bool(run.get("git_dirty")), run.get("model") or "", params


def inputs_key(run: Row) -> str:
    return json.dumps(run.get("inputs"), sort_keys=True, default=str)


def groups(runs: list[Row]) -> dict[str, list[Row]]:
    grouped: dict[tuple[str, bool, str, str], list[Row]] = defaultdict(list)
    for run in sorted(runs, key=lambda run: run.get("started_at") or ""):
        grouped[version_key(run)].append(run)
    labels: Counter[str] = Counter()
    named: dict[str, list[Row]] = {}
    for (commit, dirty, _, _), members in grouped.items():
        base = (commit[:7] or "no-git") + ("*" if dirty else "")
        labels[base] += 1
        named[base if labels[base] == 1 else f"{base} #{labels[base]}"] = members
    return named


def _avg(values: list[Any]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _diff(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def _run_quality(run: Row, plan: Row, scores: Scores) -> float | None:
    return _avg([(scores.get((run["id"], check["id"])) or {}).get("score") for check in plan["checks"]])


def _per_input(members: list[Row], quality: dict[str, float | None]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for run in members:
        if (value := quality.get(run["id"])) is not None:
            values[inputs_key(run)].append(value)
    return {key: mean(found) for key, found in values.items()}


def _compare(old: list[Row], new: list[Row], quality: dict[str, float | None]) -> Row | None:
    return stats.compare(
        _per_input(old, quality),
        _per_input(new, quality),
        [value for run in old if (value := quality.get(run["id"])) is not None],
        [value for run in new if (value := quality.get(run["id"])) is not None],
    )


def _criteria(members: list[Row], plan: Row, scores: Scores) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for run in members:
        for check in plan["checks"]:
            for name, value in (scores.get((run["id"], check["id"])) or {}).get("criteria", {}).items():
                values[name].append(value)
    return {name: mean(found) for name, found in values.items()}


def _version(label: str, members: list[Row], plan: Row, scores: Scores) -> Row:
    first = members[0]
    commit = first.get("git_commit")
    checks = {
        check["id"]: _avg([(scores.get((run["id"], check["id"])) or {}).get("score") for run in members])
        for check in plan["checks"]
    }
    return {
        "label": label,
        "commit": commit,
        "dirty": bool(first.get("git_dirty")),
        "message": git.message(commit) if commit else None,
        "model": first.get("model"),
        "params": first.get("params") or {},
        "runs": len(members),
        "scored": sum(any((run["id"], check["id"]) in scores for check in plan["checks"]) for run in members),
        "ok_rate": sum(run.get("status") == "ok" for run in members) / len(members),
        "quality": _avg(list(checks.values())),
        "checks": checks,
        "criteria": _criteria(members, plan, scores),
        "latency_s": _avg([run.get("latency_s") for run in members]),
        "input_tokens": _avg([run.get("input_tokens") for run in members]),
        "output_tokens": _avg([run.get("output_tokens") for run in members]),
        "cost": _avg([run.get("cost") for run in members]),
    }


def _asset_hashes(run: Row) -> list[str]:
    return sorted(asset.get("sha256", "") for asset in run.get("assets") or [])


def _differs(old: list[Row], new: list[Row], read: Callable[[Row], Any]) -> bool:
    before = {inputs_key(run): read(run) for run in old}
    return any(before[key] != read(run) for run in new if (key := inputs_key(run)) in before)


def _changed(before: Row, after: Row, old: list[Row], new: list[Row]) -> list[str]:
    notes = []
    if before["commit"] != after["commit"]:
        notes.append(f"commit \"{after['message']}\"" if after["message"] else f"commit {after['label']}")
    elif before["dirty"] != after["dirty"]:
        notes.append("uncommitted edits")
    if before["model"] != after["model"]:
        notes.append(f"model {before['model']} → {after['model']}")
    for key in sorted(set(before["params"]) | set(after["params"])):
        was, now = before["params"].get(key), after["params"].get(key)
        if was != now:
            notes.append(f"{key} {was} → {now}")
    if _differs(old, new, lambda run: run.get("prompt")):
        notes.append("prompt changed for the same inputs")
    if _differs(old, new, _asset_hashes):
        notes.append("input assets changed")
    return notes


def _evidence(members: list[Row], check_id: str, scores: Scores) -> list[str]:
    rows = [scores[(run["id"], check_id)] for run in members if (run["id"], check_id) in scores]
    rows = sorted((row for row in rows if row.get("score") is not None), key=lambda row: row["score"])
    return list(dict.fromkeys(row["reason"] for row in rows if row.get("reason")))[:3]


def _deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {key: value for key in after if (value := _diff(after[key], before.get(key))) is not None}


def _side_by_side(old: list[Row], new: list[Row], pairs: list[Row]) -> Row | None:
    old_ids, new_ids = {run["id"] for run in old}, {run["id"] for run in new}
    rows = [row for row in pairs if row["before_run"] in old_ids and row["after_run"] in new_ids]
    if not rows:
        return None
    wins = Counter(row["winner"] for row in rows)
    return {
        "new": wins["after"],
        "old": wins["before"],
        "tie": wins["tie"],
        "reasons": list(dict.fromkeys(row["reason"] for row in rows if row["winner"] != "tie"))[:3],
    }


def _change(before: Row, after: Row, named: dict[str, list[Row]], scores: Scores, quality: dict[str, float | None], pairs: list[Row]) -> Row:
    old, new = named[before["label"]], named[after["label"]]
    deltas = _deltas(after["checks"], before["checks"])
    worst = min(deltas, key=lambda check_id: deltas[check_id]) if deltas else None
    return {
        "from": before["label"],
        "to": after["label"],
        "stats": _compare(old, new, quality),
        "changed": _changed(before, after, old, new),
        "checks": deltas,
        "criteria": _deltas(after["criteria"], before["criteria"]),
        "worst_check": worst,
        "evidence": _evidence(new, worst, scores) if worst and deltas[worst] < -THRESHOLD else [],
        "side_by_side": _side_by_side(old, new, pairs),
        "cost_before": before["cost"],
        "cost_after": after["cost"],
    }


def _hotspots(members: list[Row]) -> list[Row]:
    totals: dict[tuple[str, str | None], Row] = defaultdict(
        lambda: {"calls": 0, "repeated": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "source": None}
    )
    for run in members:
        seen: Counter[tuple[str, str]] = Counter()
        for step in run.get("steps") or []:
            if step.get("input_tokens") is None and step.get("cost") is None:
                continue
            total = totals[(step["name"], step.get("model"))]
            signature = (step["name"], json.dumps(step.get("inputs"), sort_keys=True, default=str))
            seen[signature] += 1
            total["calls"] += 1
            total["repeated"] += int(seen[signature] > 1)
            total["input_tokens"] += step.get("input_tokens") or 0
            total["output_tokens"] += step.get("output_tokens") or 0
            total["cost"] += step.get("cost") or 0.0
            total["source"] = step.get("source")
    spend = sum(total["cost"] for total in totals.values())
    rows = [
        {
            "name": name,
            "model": model,
            **total,
            "calls_per_run": total["calls"] / len(members),
            "input_per_call": total["input_tokens"] / total["calls"],
            "output_per_call": total["output_tokens"] / total["calls"],
            "cost_per_run": total["cost"] / len(members),
            "share": total["cost"] / spend if spend else None,
        }
        for (name, model), total in totals.items()
    ]
    rows.sort(key=lambda row: (row["cost"], row["input_tokens"]), reverse=True)
    return rows[:5]


def build(plan: Row, runs: list[Row], evals: list[Row], pairs: list[Row] | None = None) -> Row:
    plan_evals = [row for row in evals if row.get("plan_id") == plan["id"]]
    plan_pairs = [row for row in pairs or [] if row.get("plan_id") == plan["id"]]
    scores = {(row["run_id"], row["check_id"]): row for row in plan_evals}
    quality = {run["id"]: _run_quality(run, plan, scores) for run in runs}
    named = groups(runs)
    versions = [_version(label, members, plan, scores) for label, members in named.items()]
    changes = [_change(before, after, named, scores, quality, plan_pairs) for before, after in pairwise(versions)]
    rated = [version for version in versions if version["quality"] is not None]
    first, last = (rated[0], rated[-1]) if len(rated) > 1 else (None, None)
    latest = named[versions[-1]["label"]] if versions else []
    return {
        "goal": plan["goal"],
        "plan_id": plan["id"],
        "checks": [{"id": check["id"], "label": describe(check)} for check in plan["checks"]],
        "versions": versions,
        "changes": changes,
        "overall": {
            "from": first["quality"] if first else None,
            "to": last["quality"] if last else None,
            "stats": _compare(named[first["label"]], named[last["label"]], quality) if first and last else None,
        },
        "hotspots": _hotspots(latest) if latest else [],
        "failures": dict(Counter((run.get("error") or {}).get("type", "Error") for run in latest if run.get("status") == "error")),
        "totals": {
            "runs": len(runs),
            "app_cost": sum(run.get("cost") or 0.0 for run in runs),
            "eval_cost": sum(row.get("cost") or 0.0 for row in plan_evals + plan_pairs),
        },
    }


def _score(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def _money(value: float | None) -> str:
    if value is None:
        return "–"
    return f"${value:.4f}" if value < 0.1 else f"${value:.2f}"


def _tokens(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def _arrow(delta: float | None) -> str:
    if delta is None or abs(delta) <= THRESHOLD:
        return ""
    return "↑" if delta > 0 else "↓"


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _verdict(result: Row | None) -> str:
    if result is None:
        return "NOT SCORED"
    interval = f"{result['delta']:+.2f} [{result['low']:+.2f}, {result['high']:+.2f}]"
    basis = f"{result['n']} matched inputs" if result["paired"] else f"different inputs, {result['n']}+ runs each"
    return f"{result['verdict'].upper()}  {interval}  ({basis})"


def _versions_table(report: Row) -> list[str]:
    check_ids = [check["id"] for check in report["checks"]]
    rows = [["version", "runs", "ok", "quality", *check_ids, "latency", "tokens/run", "cost/run"]]
    previous: Row | None = None
    for version in report["versions"]:
        cells = [
            version["label"],
            f"{version['scored']}/{version['runs']}" if version["scored"] < version["runs"] else str(version["runs"]),
            f"{version['ok_rate']:.0%}",
            _score(version["quality"]) + (_arrow(_diff(version["quality"], previous["quality"])) if previous else ""),
        ]
        for check_id in check_ids:
            value = version["checks"][check_id]
            cells.append(_score(value) + _arrow(_diff(value, previous["checks"].get(check_id) if previous else None)))
        tokens = None if version["input_tokens"] is None else version["input_tokens"] + (version["output_tokens"] or 0)
        latency = "–" if version["latency_s"] is None else f"{version['latency_s']:.1f}s"
        rows.append([*cells, latency, _tokens(tokens), _money(version["cost"])])
        previous = version
    return _table(rows)


def _change_lines(change: Row, labels: dict[str, str]) -> list[str]:
    what = ", ".join(change["changed"]) or "no tracked difference"
    lines = [f"  {change['from']} → {change['to']}  {_verdict(change['stats'])}", f"      changed: {what}"]
    worst = change["worst_check"]
    if worst and change["checks"][worst] < -THRESHOLD:
        lines.append(f"      biggest drop: {labels[worst]} {change['checks'][worst]:+.2f}")
    lines += [f"      rubric: {name} {value:+.2f}" for name, value in change["criteria"].items() if value < -THRESHOLD]
    lines += [f"      judge: \"{reason}\"" for reason in change["evidence"]]
    side = change["side_by_side"]
    if side:
        lines.append(f"      side by side: new better {side['new']} · old better {side['old']} · tie {side['tie']}")
        lines += [f"        \"{reason}\"" for reason in side["reasons"]]
    before, after = change["cost_before"], change["cost_after"]
    if before and after and abs(after - before) / before > 0.1:
        lines.append(f"      cost/run {_money(before)} → {_money(after)}")
    return lines


def _hotspot_lines(report: Row) -> list[str]:
    rows = [["call", "model", "calls/run", "in/call", "out/call", "cost/run", "share"]]
    for spot in report["hotspots"]:
        rows.append(
            [
                spot["name"] + (f" ({spot['repeated']} repeated)" if spot["repeated"] else ""),
                spot["model"] or "–",
                f"{spot['calls_per_run']:.1f}",
                _tokens(spot["input_per_call"]),
                _tokens(spot["output_per_call"]),
                _money(spot["cost_per_run"]),
                "–" if spot["share"] is None else f"{spot['share']:.0%}",
            ]
        )
    return _table(rows)


def render(report: Row) -> str:
    versions = report["versions"]
    lines = [f"Goal: {report['goal']}", "", "Checks"]
    lines += [f"  {check['id']}  {check['label']}" for check in report["checks"]]
    if not versions:
        return "\n".join(lines + ["", "No runs yet."])
    lines += ["", "Versions (oldest first)", *_versions_table(report)]

    names = list(dict.fromkeys(name for version in versions for name in version["criteria"]))
    if names:
        rows = [["version", *names]]
        rows += [[version["label"], *(_score(version["criteria"].get(name)) for name in names)] for version in versions]
        lines += ["", "Against your references", *_table(rows)]

    overall = report["overall"]
    if overall["stats"]:
        lines += ["", f"Overall {overall['from']:.2f} → {overall['to']:.2f}: {_verdict(overall['stats'])}"]

    if report["changes"]:
        labels = {check["id"]: check["label"] for check in report["checks"]}
        lines += ["", "What changed"]
        for change in report["changes"]:
            lines += _change_lines(change, labels)

    if report["hotspots"]:
        lines += ["", f"Where the tokens go ({versions[-1]['label']})", *_hotspot_lines(report)]

    if report["failures"]:
        lines += ["", f"Failures ({versions[-1]['label']})"]
        lines += [f"  {kind} ×{count}" for kind, count in report["failures"].items()]

    totals = report["totals"]
    lines += [
        "",
        f"{totals['runs']} runs · app spend {_money(totals['app_cost'])} · eval spend {_money(totals['eval_cost'])}",
        "Verdicts use a 95% bootstrap interval. Run `lens suggest` for exact fixes.",
    ]
    return "\n".join(lines)
