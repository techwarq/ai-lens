from ai_lens.models import Run, Step
from ai_lens.store import data_dir, load_plan, load_runs, save_plan, save_run


def test_round_trip_with_steps():
    save_run(Run(id="r1", name="app", steps=[Step(id="s1", name="upscale")]))
    [run] = load_runs()
    assert run["id"] == "r1"
    assert run["steps"][0]["name"] == "upscale"


def test_broken_lines_are_skipped():
    save_run(Run(id="r1", name="app"))
    with open(data_dir() / "runs.jsonl", "a", encoding="utf-8") as file:
        file.write('{"id": "half\n')
    save_run(Run(id="r2", name="app"))
    assert [run["id"] for run in load_runs()] == ["r1", "r2"]


def test_plan_round_trip():
    assert load_plan() is None
    save_plan({"id": "p1", "goal": "g", "checks": []})
    assert load_plan() == {"id": "p1", "goal": "g", "checks": []}
