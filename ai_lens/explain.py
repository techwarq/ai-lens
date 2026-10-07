import hashlib
import json

from . import llm, store
from .report import Row

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

PROMPT = """You are explaining an eval result to the developer of an AI application, who will read this before the scores.

What they track:
{goal}

The verdict for their latest change is {verdict}. The evidence:
{facts}

Write 2 or 3 short sentences in plain, everyday words.
- Start with what went wrong, or what got better if nothing went wrong.
- Say what the outputs now do differently, using what the judge and the side-by-side comparisons saw. Quote one concrete example if there is one.
- Mention what changed in the code if it is known.
- Use at most one number. The scores are shown right below your text.
- No advice, no hedging, no headings."""

EXPLAINED = ("better", "worse", "mixed")


def facts(report: Row, change: Row) -> Row:
    names = {check["id"]: check["name"] for check in report["checks"]}
    return {
        "what changed in the code": change["changed"],
        "checks that moved": [
            {
                "check": names[move["check"]],
                "before": round(move["before"], 2),
                "after": round(move["after"], 2),
                "beyond noise": move["verdict"] in ("better", "worse"),
                "the judge said about the new version": move["evidence"],
            }
            for move in change["moves"]
            if move["verdict"] in ("better", "worse")
        ],
        "pass/fail checks that started failing": [
            {"check": names[gate["check"]], "passed before": gate["before"], "passed now": gate["after"]}
            for gate in change["gates"]
            if gate["broken"]
        ],
        "failure reasons": change["evidence"] if change["verdict"]["broken"] else [],
        "side by side": change["side_by_side"],
        "cost per run": {"before": change["cost_before"], "after": change["cost_after"]},
    }


def explain(report: Row, ask: bool = True) -> float:
    if not report["changes"]:
        return 0.0
    change = report["changes"][-1]
    if change["verdict"]["label"] not in EXPLAINED:
        return 0.0
    text = json.dumps(facts(report, change), indent=1, default=str)
    prompt = PROMPT.format(goal=report["goal"], verdict=change["verdict"]["label"].upper(), facts=text)
    key = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    cached = next((row["summary"] for row in store.load_summaries() if row.get("key") == key), None)
    if cached or not ask:
        change["summary"] = cached
        return 0.0
    try:
        data, cost = llm.ask_json([prompt], SCHEMA)
    except llm.LensError:
        return 0.0
    change["summary"] = str(data["summary"]).strip()
    store.save_summaries([{"key": key, "summary": change["summary"]}])
    return cost
