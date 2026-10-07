import json
import textwrap
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise
from statistics import mean
from typing import Any

from . import git, stats
from .evaluators import EVALUATORS
from .plan import describe

THRESHOLD = 0.05
LOPSIDED = 0.8

Row = dict[str, Any]
Scores = dict[tuple[str, str], Row]
Score = Callable[[Row], float | None]
VersionKey = tuple[str, str, str, str]


def _edits(run: Row) -> str:
    if "git_edits" in run:
        return run["git_edits"] or ""
    return "dirty" if run.get("git_dirty") else ""


def version_key(run: Row) -> VersionKey:
    params = json.dumps(run.get("params") or {}, sort_keys=True, default=str)
    return run.get("git_commit") or "", _edits(run), run.get("model") or "", params


def inputs_key(run: Row) -> str:
    return json.dumps(run.get("inputs"), sort_keys=True, default=str)


def _base_label(commit: str, edits: str) -> str:
    label = commit[:7] or "no-git"
    if edits == "dirty":
        return f"{label} +edits"
    return f"{label} +edit {edits[:6]}" if edits else label


def _difference(first: VersionKey, key: VersionKey) -> str:
    if first[2] != key[2]:
        return key[2] or "no model"
    was, now = json.loads(first[3]), json.loads(key[3])
    return ", ".join(f"{name}={now.get(name)}" for name in sorted(set(was) | set(now)) if was.get(name) != now.get(name))


def groups(runs: list[Row]) -> dict[str, list[Row]]:
    grouped: dict[VersionKey, list[Row]] = defaultdict(list)
    for run in sorted(runs, key=lambda run: run.get("started_at") or ""):
        grouped[version_key(run)].append(run)
    firsts: dict[str, VersionKey] = {}
    named: dict[str, list[Row]] = {}
    for key, members in grouped.items():
        base = _base_label(key[0], key[1])
        first = firsts.setdefault(base, key)
        label = base if first == key else f"{base} · {_difference(first, key)}"
        unique, count = label, 1
        while unique in named:
            count += 1
            unique = f"{label} #{count}"
        named[unique] = members
    return named


def is_gate(check: Row) -> bool:
    evaluator = EVALUATORS.get(check["evaluator"])
    return bool(evaluator and evaluator.gate)


def _avg(values: list[Any]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _diff(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def _score_of(scores: Scores, check_id: str) -> Score:
    return lambda run: (scores.get((run["id"], check_id)) or {}).get("score")


def _present(members: list[Row], score: Score) -> list[float]:
    return [value for run in members if (value := score(run)) is not None]


def _per_input(members: list[Row], score: Score) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for run in members:
        if (value := score(run)) is not None:
            values[inputs_key(run)].append(value)
    return {key: mean(found) for key, found in values.items()}


def _compare(old: list[Row], new: list[Row], score: Score) -> Row | None:
    return stats.compare(_per_input(old, score), _per_input(new, score), _present(old, score), _present(new, score))


def _quality_checks(plan: Row) -> list[Row]:
    return [check for check in plan["checks"] if not is_gate(check)] or plan["checks"]


def _run_quality(run: Row, plan: Row, scores: Scores) -> float | None:
    return _avg([_score_of(scores, check["id"])(run) for check in _quality_checks(plan)])


def _passed(run: Row, plan: Row, scores: Scores) -> bool | None:
    found = [_score_of(scores, check["id"])(run) for check in plan["checks"] if is_gate(check)]
    present = [value for value in found if value is not None]
    return all(value >= 1 for value in present) if present else None


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
    passed = [result for run in members if (result := _passed(run, plan, scores)) is not None]
    return {
        "label": label,
        "commit": commit,
        "dirty": bool(first.get("git_dirty")),
        "edits": _edits(first) or None,
        "edited": first.get("git_edited") or [],
        "message": git.message(commit) if commit else None,
        "model": first.get("model"),
        "params": first.get("params") or {},
        "runs": len(members),
        "scored": sum(any((run["id"], check["id"]) in scores for check in plan["checks"]) for run in members),
        "ok_rate": sum(run.get("status") == "ok" for run in members) / len(members),
        "pass_rate": sum(passed) / len(passed) if passed else None,
        "quality": _avg([_run_quality(run, plan, scores) for run in members]),
        "checks": {check["id"]: _avg(_present(members, _score_of(scores, check["id"]))) for check in plan["checks"]},
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


def _edit_note(before: Row, after: Row) -> str | None:
    if before["edits"] == after["edits"]:
        return None
    files = f" to {', '.join(after['edited'][:3])}" if after["edited"] else ""
    if not after["edits"]:
        return "uncommitted edits removed"
    return f"{'different ' if before['edits'] else ''}uncommitted edits{files}"


def _changed(before: Row, after: Row, old: list[Row], new: list[Row]) -> list[str]:
    notes = []
    if before["commit"] != after["commit"]:
        notes.append(f"commit \"{after['message']}\"" if after["message"] else f"commit {(after['commit'] or 'none')[:7]}")
    if note := _edit_note(before, after):
        notes.append(note)
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


def _evidence(members: list[Row], check_id: str, scores: Scores, best: bool = False) -> list[str]:
    rows = [scores[(run["id"], check_id)] for run in members if (run["id"], check_id) in scores]
    rows = sorted((row for row in rows if row.get("score") is not None), key=lambda row: row["score"], reverse=best)
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


def _lean(side: Row | None) -> str | None:
    if side is None:
        return None
    decisive = side["new"] + side["old"]
    if decisive < stats.MIN_RUNS or decisive < side["tie"]:
        return None
    if side["new"] >= LOPSIDED * decisive:
        return "better"
    if side["old"] >= LOPSIDED * decisive:
        return "worse"
    return None


def _moves(old: list[Row], new: list[Row], plan: Row, scores: Scores) -> list[Row]:
    moves = []
    for check in plan["checks"]:
        score = _score_of(scores, check["id"])
        result = None if is_gate(check) else _compare(old, new, score)
        if result:
            evidence = _evidence(new, check["id"], scores, best=result["delta"] > 0)
            moves.append({"check": check["id"], "before": _avg(_present(old, score)), "after": _avg(_present(new, score)), **result, "evidence": evidence})
    return moves


def _gates(old: list[Row], new: list[Row], plan: Row, scores: Scores) -> list[Row]:
    gates = []
    for check in plan["checks"]:
        score = _score_of(scores, check["id"])
        before, after = _avg(_present(old, score)), _avg(_present(new, score))
        if is_gate(check) and before is not None and after is not None:
            gates.append({"check": check["id"], "before": before, "after": after, "broken": after < before - THRESHOLD})
    return gates


def _verdict(moves: list[Row], gates: list[Row], side: Row | None, overall: Row | None) -> Row:
    ups = [move["check"] for move in moves if move["verdict"] == "better"]
    downs = [move["check"] for move in moves if move["verdict"] == "worse"]
    broken = [gate["check"] for gate in gates if gate["broken"]]
    lean = _lean(side)
    if broken:
        label = "worse"
    elif lean:
        label = lean
    elif ups and downs:
        label = "mixed"
    elif downs:
        label = "worse"
    elif ups:
        label = "better"
    elif not moves and not gates and not side:
        label = "not scored"
    elif moves and all(move["verdict"] == "not enough data" for move in moves):
        label = "not enough data"
    else:
        label = "no clear change"
    need = stats.inputs_needed(overall) if label in ("no clear change", "not enough data") else None
    return {"label": label, "ups": ups, "downs": downs, "broken": broken, "side": lean, "need": need}


def _assess(old: list[Row], new: list[Row], plan: Row, scores: Scores, quality: dict[str, float | None], side: Row | None = None) -> Row:
    moves, gates = _moves(old, new, plan, scores), _gates(old, new, plan, scores)
    overall = _compare(old, new, lambda run: quality.get(run["id"]))
    return {"stats": overall, "moves": moves, "gates": gates, "verdict": _verdict(moves, gates, side, overall)}


def _worst(assessment: Row) -> str | None:
    broken = assessment["verdict"]["broken"]
    if broken:
        return broken[0]
    dropped = [move for move in assessment["moves"] if move["delta"] < -THRESHOLD]
    return min(dropped, key=lambda move: move["delta"])["check"] if dropped else None


def _change(before: Row, after: Row, named: dict[str, list[Row]], plan: Row, scores: Scores, quality: dict[str, float | None], pairs: list[Row]) -> Row:
    old, new = named[before["label"]], named[after["label"]]
    side = _side_by_side(old, new, pairs)
    assessment = _assess(old, new, plan, scores, quality, side)
    worst = _worst(assessment)
    return {
        "from": before["label"],
        "to": after["label"],
        **assessment,
        "changed": _changed(before, after, old, new),
        "criteria": _deltas(after["criteria"], before["criteria"]),
        "worst_check": worst,
        "evidence": _evidence(new, worst, scores) if worst else [],
        "side_by_side": side,
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
    changes = [_change(before, after, named, plan, scores, quality, plan_pairs) for before, after in pairwise(versions)]
    first, last = (versions[0], versions[-1]) if len(versions) > 1 else (None, None)
    latest = named[versions[-1]["label"]] if versions else []
    return {
        "goal": plan["goal"],
        "plan_id": plan["id"],
        "checks": [
            {"id": check["id"], "label": describe(check), "name": check.get("check") or describe(check), "gate": is_gate(check)}
            for check in plan["checks"]
        ],
        "versions": versions,
        "changes": changes,
        "overall": {"from": first["label"], "to": last["label"], **_assess(named[first["label"]], named[last["label"]], plan, scores, quality)}
        if first and last
        else None,
        "hotspots": _hotspots(latest) if latest else [],
        "failures": dict(Counter((run.get("error") or {}).get("type", "Error") for run in latest if run.get("status") == "error")),
        "totals": {
            "runs": len(runs),
            "app_cost": sum(run.get("cost") or 0.0 for run in runs),
            "eval_cost": sum(row.get("cost") or 0.0 for row in plan_evals + plan_pairs),
        },
    }


STYLES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33"}
COLORS = {"better": "green", "worse": "red", "mixed": "yellow"}
Paint = Callable[[str, str], str]


def _painter(color: bool) -> Paint:
    return lambda style, text: f"\033[{STYLES[style]}m{text}\033[0m" if color and text else text


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


def _short(text: str, limit: int = 44) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def _spread(move: Row) -> str:
    return f"±{(move['high'] - move['low']) / 2:.2f}"


def _basis(result: Row | None) -> str:
    if result is None:
        return ""
    return f"{result['n']} matched inputs" if result["paired"] else f"different inputs, {result['n']}+ runs each"


def _move_text(move: Row) -> str:
    return f"{move['check']} {_arrow(move['delta']) or '→'}{abs(move['delta']):.2f}"


def _reason(change: Row) -> str:
    verdict = change["verdict"]
    moves = {move["check"]: move for move in change["moves"]}
    gates = {gate["check"]: gate for gate in change["gates"]}
    moved = ", ".join(_move_text(moves[check]) for check in verdict["downs"] + verdict["ups"])
    if verdict["broken"]:
        gate = gates[verdict["broken"][0]]
        return f"{gate['check']} passed {gate['before']:.0%} → {gate['after']:.0%}"
    if verdict["side"]:
        side = change["side_by_side"]
        winner, loser = (side["new"], side["old"]) if verdict["side"] == "better" else (side["old"], side["new"])
        prefers = "new" if verdict["side"] == "better" else "old"
        return f"side by side prefers the {prefers} version {winner}–{loser}" + (f" · {moved}" if moved else "")
    if moved:
        return moved
    if verdict["label"] == "not scored":
        return "no scores yet"
    if verdict["label"] == "not enough data":
        return f"need at least {stats.MIN_RUNS} matched inputs to compare"
    return "no check moved beyond noise"


OPENERS = {
    "better": "The new version is better.",
    "worse": "The new version is worse.",
    "mixed": "Mixed result: some things improved and others got worse.",
    "no clear change": "Nothing changed beyond noise.",
    "not enough data": "Not enough runs to compare yet.",
    "not scored": "These runs haven't been scored yet.",
}


def story(change: Row, names: dict[str, str]) -> str:
    verdict = change["verdict"]
    moves = {move["check"]: move for move in change["moves"]}
    gates = {gate["check"]: gate for gate in change["gates"]}
    sentences = [OPENERS[verdict["label"]]]
    for check in verdict["broken"]:
        gate = gates[check]
        sentences.append(f"\"{names[check]}\" started failing: {gate['before']:.0%} passed before, {gate['after']:.0%} now.")
    for label, found in (("Got worse", verdict["downs"]), ("Got better", verdict["ups"])):
        if found:
            listed = "; ".join(f"{names[check]} ({moves[check]['before']:.2f} → {moves[check]['after']:.2f})" for check in found)
            sentences.append(f"{label}: {listed}.")
    side = change.get("side_by_side")
    if verdict["side"] and side:
        winner = "new" if verdict["side"] == "better" else "old"
        sentences.append(f"Side by side, the {winner} version won {side[winner]} of {side['new'] + side['old'] + side['tie']}.")
    return " ".join(sentences)


def _wrap(text: str, indent: str = "    ", width: int = 92) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _need(change: Row) -> str | None:
    need, result = change["verdict"]["need"], change["stats"]
    if need is None or result is None:
        return None
    if need == 0:
        return f"no change bigger than {stats.EFFECT:.2f} on {_basis(result)}"
    return f"add about {need} more matched inputs to tell a {stats.EFFECT:.2f} change from noise"


def _headline(change: Row, paint: Paint, width: int = 0) -> str:
    label = change["verdict"]["label"]
    return paint(COLORS.get(label, "dim"), paint("bold", label.upper())) + " " * (width - len(label))


def _check_lines(change: Row, names: dict[str, str], paint: Paint) -> list[str]:
    rows: list[tuple[str, str, str, str]] = []
    for gate in change["gates"]:
        if gate["broken"]:
            rows.append(("✗", gate["check"], f"passed {gate['before']:.0%} → {gate['after']:.0%}", ""))
    for move in sorted(change["moves"], key=lambda move: move["delta"]):
        if abs(move["delta"]) <= THRESHOLD and move["verdict"] not in ("better", "worse"):
            continue
        sure = move["verdict"] in ("better", "worse")
        values = f"{_score(move['before'])} → {_score(move['after'])}  {move['delta']:+.2f} {_spread(move)}"
        rows.append((_arrow(move["delta"]) if sure else "~", move["check"], values, "" if sure else "could be noise"))
    width = max((len(_short(names[row[1]])) for row in rows), default=0)
    quotes = {move["check"]: move["evidence"][:1] for move in change["moves"] if move["verdict"] in ("better", "worse")}
    lines = []
    for mark, check, values, note in rows:
        style = "red" if mark in ("✗", "↓") else "green" if mark == "↑" else "dim"
        text = f"    {paint(style, mark)} {check:<4}{_short(names[check]):<{width}}  {values}"
        lines.append(text + (f"  {paint('dim', note)}" if note else ""))
        lines += [f"           {paint('dim', chr(34) + _short(quote, 100) + chr(34))}" for quote in quotes.get(check, [])]
    return lines


def _change_lines(change: Row, names: dict[str, str], paint: Paint) -> list[str]:
    lines = [f"  {_headline(change, paint)}  {change['from']} → {change['to']}", ""]
    lines += _wrap(change.get("summary") or story(change, names))
    lines += ["", "    Scores", *_check_lines(change, names, paint)] if change["moves"] or change["gates"] else []
    lines += [f"      rubric: {name} {value:+.2f}" for name, value in change["criteria"].items() if value < -THRESHOLD]
    side = change["side_by_side"]
    if side:
        lines += ["", f"    side by side  new better {side['new']} · old better {side['old']} · tie {side['tie']}"]
        lines += [f"      {paint('dim', chr(34) + _short(reason, 110) + chr(34))}" for reason in side["reasons"]]
    lines += ["", f"    changed  {', '.join(change['changed']) or 'nothing ai-lens can see'}"]
    before, after = change["cost_before"], change["cost_after"]
    if before and after and abs(after - before) / before > 0.1:
        lines.append(f"    cost/run {_money(before)} → {_money(after)}")
    basis, need = _basis(change["stats"]), _need(change)
    if basis or need:
        lines.append(f"    {paint('dim', ' · '.join(part for part in (basis, need) if part))}")
    return lines


def _summary_line(change: Row, paint: Paint, width: int, label_width: int) -> str:
    changed = f"  ({', '.join(change['changed'])})" if change.get("changed") else ""
    line = f"  {change['from']} → {change['to']}".ljust(width) + f"  {_headline(change, paint, label_width)}  {_reason(change)}"
    return line + paint("dim", changed)


def _versions_table(report: Row) -> list[str]:
    rated = [check["id"] for check in report["checks"] if not check["gate"]]
    rows = [["version", "runs", "pass", *rated, "cost/run", "tokens/run", "time"]]
    previous: Row | None = None
    for version in report["versions"]:
        cells = [
            version["label"],
            f"{version['scored']}/{version['runs']}" if version["scored"] < version["runs"] else str(version["runs"]),
            "–" if version["pass_rate"] is None else f"{version['pass_rate']:.0%}",
        ]
        for check_id in rated:
            value = version["checks"][check_id]
            cells.append(_score(value) + _arrow(_diff(value, previous["checks"].get(check_id) if previous else None)))
        tokens = None if version["input_tokens"] is None else version["input_tokens"] + (version["output_tokens"] or 0)
        latency = "–" if version["latency_s"] is None else f"{version['latency_s']:.1f}s"
        rows.append([*cells, _money(version["cost"]), _tokens(tokens), latency])
        previous = version
    return _table(rows)


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


def render(report: Row, color: bool = False) -> str:
    paint = _painter(color)
    versions, changes = report["versions"], report["changes"]
    names = {check["id"]: check["name"] for check in report["checks"]}
    lines = [paint("bold", report["goal"])]
    if not versions:
        return "\n".join(lines + ["", "No runs yet."])

    if changes:
        lines += ["", paint("bold", "Latest change"), *_change_lines(changes[-1], names, paint)]
    else:
        lines += ["", f"One version so far ({versions[0]['label']}). Change something and run again to compare."]

    lines += ["", paint("bold", "Versions"), *_versions_table(report)]

    overall = [report["overall"]] if len(versions) > 2 else []
    history = [*changes[:-1], *overall]
    if history:
        width = max(len(f"  {change['from']} → {change['to']}") for change in history)
        label_width = max(len(change["verdict"]["label"]) for change in history)
        lines += ["", paint("bold", "History")]
        lines += [_summary_line(change, paint, width, label_width) for change in changes[:-1]]
        lines += [_summary_line(change, paint, width, label_width) + paint("dim", "  (first → latest)") for change in overall]

    criteria = list(dict.fromkeys(name for version in versions for name in version["criteria"]))
    if criteria:
        rows = [["version", *criteria]]
        rows += [[version["label"], *(_score(version["criteria"].get(name)) for name in criteria)] for version in versions]
        lines += ["", paint("bold", "Against your references"), *_table(rows)]

    if report["hotspots"]:
        lines += ["", paint("bold", f"Where the tokens go ({versions[-1]['label']})"), *_hotspot_lines(report)]

    if report["failures"]:
        lines += ["", paint("bold", f"Failures ({versions[-1]['label']})")]
        lines += [f"  {kind} ×{count}" for kind, count in report["failures"].items()]

    lines += ["", paint("bold", "Checks")]
    lines += [f"  {check['id']:<4}{check['label']}" + (paint("dim", "  (pass/fail)") if check["gate"] else "") for check in report["checks"]]

    totals = report["totals"]
    lines += [
        "",
        paint(
            "dim",
            f"{totals['runs']} runs · app {_money(totals['app_cost'])} · evals {_money(totals['eval_cost'])} · "
            "± is the 95% range · `lens suggest` for exact fixes",
        ),
    ]
    return "\n".join(lines)
