import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

from . import llm, pairwise, references, report, store, suggest
from .explain import explain
from .evaluate import evaluate
from .llm import LensError
from .media import OUTPUT_KINDS
from .plan import describe, make_plan


def _plan() -> dict[str, Any]:
    plan = store.load_plan()
    if plan is None:
        raise LensError('Nothing is tracked yet. Start with: lens track "what you want to know"')
    return plan


def _runs() -> list[dict[str, Any]]:
    runs = store.load_runs()
    if not runs:
        raise LensError("No runs recorded yet. Add @trace to your app's main function and run it.")
    return runs


def track(args: argparse.Namespace) -> int:
    if args.model:
        llm.set_model(args.model)
    plan = make_plan(args.goal, args.kind)
    print(f"Tracking: {plan['goal']}")
    print(f"Output type: {plan['kind']}")
    for check in plan["checks"]:
        print(f"  {check['id']}  {describe(check)}")
        for item in check.get("rubric", []):
            print(f"        - {item['name']}: {item['description']}")
    if plan["planner"] == "default":
        print("\nUsed the default checks because no model could be reached. Edit .ai-lens/plan.json to change them.")
        try:
            llm.load_model()
        except LensError as error:
            print(error)
    print("\nRun your app as usual, then: lens inspect")
    return 0


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _build(plan: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return report.build(plan, runs, store.load_evals(), store.load_pairs())


def inspect(args: argparse.Namespace) -> int:
    plan, runs = _plan(), _runs()
    if plan.get("references") != references.fingerprint(references.load()):
        _warn("your references changed since `lens track`; run it again to rebuild the rubric.")
    if plan.get("judge") != llm.location():
        _warn("your model changed since `lens track`, so new scores aren't comparable with old ones; run `lens track` again.")
    if not args.no_eval:
        rows, errors = evaluate(plan, runs, per_version=args.per_version)
        pairs, pair_errors = pairwise.run(plan, runs, per_change=args.pairs) if args.pairs else ([], [])
        spent = sum(row.get("cost") or 0.0 for row in rows + pairs)
        if rows or pairs:
            print(f"Scored {len(rows)} checks and {len(pairs)} side-by-side comparisons (${spent:.4f}).", file=sys.stderr)
        for error, count in Counter(errors + pair_errors).most_common(3):
            _warn(f"{error} (×{count})")
        print(file=sys.stderr)
    built = _build(plan, runs)
    explain(built, ask=not args.no_eval)
    print(json.dumps(built, default=str, indent=2) if args.json else report.render(built, color=_color()))
    return 0


def advise(args: argparse.Namespace) -> int:
    plan, runs = _plan(), _runs()
    built = _build(plan, runs)
    print("Reading the report and your code…\n", file=sys.stderr)
    result, cost = suggest.suggest(built, runs)
    print(suggest.render(result))
    print(f"\n(suggestions cost ${cost:.4f})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lens", description="Track whether your AI app gets better or worse with every change.")
    commands = parser.add_subparsers(dest="command", required=True)

    track_parser = commands.add_parser("track", help="say what to track; creates the evaluation plan")
    track_parser.add_argument("goal", help='e.g. "are my videos getting better with my changes?"')
    track_parser.add_argument("--type", dest="kind", choices=OUTPUT_KINDS, help="output type, detected from runs if omitted")
    track_parser.add_argument("--model", help="your model function as module:function, used for planning and judging")
    track_parser.set_defaults(handler=track)

    inspect_parser = commands.add_parser("inspect", help="score new runs and show the report")
    inspect_parser.add_argument("--no-eval", action="store_true", help="only show the report, don't score new runs")
    inspect_parser.add_argument("--per-version", type=int, default=20, help="max runs to score per version (default 20)")
    inspect_parser.add_argument("--pairs", type=int, default=5, help="side-by-side comparisons per change (default 5, 0 to skip)")
    inspect_parser.add_argument("--json", action="store_true", help="print the report as JSON")
    inspect_parser.set_defaults(handler=inspect)

    suggest_parser = commands.add_parser("suggest", help="suggest exact code changes for regressions and token waste")
    suggest_parser.set_defaults(handler=advise)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except LensError as error:
        print(f"lens: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
