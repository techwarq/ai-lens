import pytest

from ai_lens.report import build, render

PLAN = {
    "id": "p1",
    "goal": "are my videos getting better?",
    "checks": [
        {"id": "c1", "evaluator": "success", "check": ""},
        {"id": "c2", "evaluator": "judge", "check": "Smooth motion?"},
    ],
}

LLM_STEP = {"name": "write_script", "model": "claude-sonnet-5-5", "input_tokens": 5000, "output_tokens": 500, "cost": 0.015, "inputs": {"q": "same"}}


def make_run(run_id, commit, model, smooth, at, topic=0):
    run = {
        "id": run_id,
        "status": "ok",
        "git_commit": commit,
        "model": model,
        "params": {"steps": 30},
        "inputs": {"topic": topic},
        "started_at": at,
        "latency_s": 2.0,
        "input_tokens": 10_000,
        "output_tokens": 1000,
        "cost": 0.03,
        "steps": [LLM_STEP, LLM_STEP],
    }
    evals = [
        {"plan_id": "p1", "run_id": run_id, "check_id": "c1", "score": 1.0, "reason": "completed"},
        {"plan_id": "p1", "run_id": run_id, "check_id": "c2", "score": smooth, "reason": f"motion {smooth:.2f}"},
    ]
    return run, evals


def scenario(topics_per_version=(range(5), range(5), range(5))):
    runs, evals = [], []
    versions = [("aaaaaaa1", "gen-v2", 0.8), ("bbbbbbb2", "gen-v3-fast", 0.4), ("ccccccc3", "gen-v2", 0.9)]
    for index, ((commit, model, smooth), topics) in enumerate(zip(versions, topics_per_version)):
        for topic in topics:
            run, rows = make_run(f"{commit}-{topic}", commit, model, smooth - topic * 0.02, f"2026-10-0{index + 1}T{topic:02d}", topic)
            runs.append(run)
            evals += rows
    return runs, evals


def test_changes_are_judged_on_matched_inputs_with_intervals():
    report = build(PLAN, *scenario())
    assert [version["label"] for version in report["versions"]] == ["aaaaaaa", "bbbbbbb", "ccccccc"]
    worse, better = report["changes"]
    assert worse["stats"]["verdict"] == "worse"
    assert worse["stats"]["paired"] is True
    assert worse["stats"]["n"] == 5
    assert worse["stats"]["high"] < 0
    assert worse["worst_check"] == "c2"
    assert "model gen-v2 → gen-v3-fast" in worse["changed"]
    assert worse["evidence"][0].startswith("motion 0.3")
    assert better["stats"]["verdict"] == "better"
    assert report["overall"]["stats"]["verdict"] == "better"


def test_different_inputs_fall_back_to_unpaired():
    report = build(PLAN, *scenario((range(5), range(5, 10), range(10, 15))))
    stats = report["changes"][0]["stats"]
    assert stats["paired"] is False
    assert stats["verdict"] == "worse"


def test_too_few_runs_is_not_a_verdict():
    report = build(PLAN, *scenario((range(2), range(2), range(2))))
    assert report["changes"][0]["stats"]["verdict"] == "not enough data"


def test_side_by_side_results_are_counted():
    runs, evals = scenario()
    pairs = [
        {"plan_id": "p1", "before_run": f"aaaaaaa1-{topic}", "after_run": f"bbbbbbb2-{topic}", "winner": winner, "reason": "old is smoother"}
        for topic, winner in enumerate(["before", "before", "before", "tie"])
    ]
    side = build(PLAN, runs, evals, pairs)["changes"][0]["side_by_side"]
    assert side == {"new": 0, "old": 3, "tie": 1, "reasons": ["old is smoother"]}


def test_hotspots_find_repeated_calls():
    [spot] = build(PLAN, *scenario())["hotspots"]
    assert spot["name"] == "write_script"
    assert spot["calls_per_run"] == 2
    assert spot["repeated"] == 5
    assert spot["share"] == 1.0


def test_render_mentions_what_broke():
    text = render(build(PLAN, *scenario()))
    assert "WORSE" in text
    assert "5 matched inputs" in text
    assert "gen-v2 → gen-v3-fast" in text
    assert "write_script" in text
    assert "lens suggest" in text


def test_empty_report():
    assert "No runs yet." in render(build(PLAN, [], []))


def test_prompt_changes_and_rubric_scores_are_reported():
    plan = {**PLAN, "checks": [*PLAN["checks"], {"id": "c3", "evaluator": "reference", "rubric": []}]}
    runs, evals = [], []
    for commit, prompt, sharp, at in [("aaaaaaa1", "a cat", 0.9, "1"), ("bbbbbbb2", "blurry cat", 0.4, "2")]:
        run, rows = make_run(commit, commit, "gen", 0.8, at)
        runs.append({**run, "prompt": prompt})
        evals += rows + [{"plan_id": "p1", "run_id": commit, "check_id": "c3", "score": sharp, "criteria": {"Sharp subject": sharp}}]
    report = build(plan, runs, evals)
    [change] = report["changes"]
    assert "prompt changed for the same inputs" in change["changed"]
    assert change["criteria"]["Sharp subject"] == pytest.approx(-0.5)
    text = render(report)
    assert "Against your references" in text
    assert "rubric: Sharp subject -0.50" in text
