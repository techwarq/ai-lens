from ai_lens import pairwise
from ai_lens.config import settings
from ai_lens.store import load_pairs

PLAN = {"id": "p1", "goal": "better captions", "checks": [{"id": "c2", "evaluator": "judge", "check": "Vivid?"}]}


def make_run(run_id, commit, topic, output):
    return {
        "id": run_id,
        "status": "ok",
        "git_commit": commit,
        "inputs": {"topic": topic},
        "output": output,
        "output_type": "text",
        "started_at": run_id,
    }


RUNS = [make_run(f"a{topic}", "aaa", topic, f"old {topic}") for topic in range(3)] + [
    make_run(f"b{topic}", "bbb", topic, f"new {topic}") for topic in range(3)
] + [make_run("b9", "bbb", 9, "unmatched")]


def prefers(word):
    def model(parts):
        a = parts[parts.index("Output A:") + 1]
        winner = "A" if a.startswith(word) else "B"
        return f'{{"winner": "{winner}", "reason": "{word} is better"}}'

    return model


def test_only_matched_inputs_are_compared_and_cached(monkeypatch):
    monkeypatch.setattr(settings, "model", prefers("new"))
    rows, errors = pairwise.run(PLAN, RUNS, per_change=2)
    assert errors == []
    assert len(rows) == 2
    assert {row["winner"] for row in rows} == {"after"}
    assert all(row["consistent"] for row in rows)
    more, _ = pairwise.run(PLAN, RUNS, per_change=3)
    assert len(more) == 1
    assert len(load_pairs()) == 3


def test_order_bias_becomes_a_tie(monkeypatch):
    monkeypatch.setattr(settings, "model", lambda parts: '{"winner": "A", "reason": "first one"}')
    rows, _ = pairwise.run(PLAN, RUNS, per_change=1)
    assert rows[0]["winner"] == "tie"
    assert rows[0]["consistent"] is False
