from ai_lens import media
from ai_lens.evaluate import evaluate
from ai_lens.evaluators import EVALUATORS
from ai_lens.store import load_evals

PLAN = {
    "id": "p1",
    "checks": [
        {"id": "c1", "evaluator": "success", "check": ""},
        {"id": "c2", "evaluator": "not_empty", "check": ""},
        {"id": "c3", "evaluator": "judge", "check": "Is it on topic?"},
    ],
}


def run(run_id: str, status: str = "ok", output: str | None = "hello", commit: str = "aaa"):
    return {
        "id": run_id,
        "status": status,
        "output": output,
        "output_type": "text",
        "inputs": {"q": "x"},
        "git_commit": commit,
        "started_at": run_id,
    }


def test_scores_every_check_once(fake_llm):
    runs = [run("r1"), run("r2", status="error", output=None)]
    rows, errors = evaluate(PLAN, runs)
    assert errors == []
    scores = {(row["run_id"], row["check_id"]): row["score"] for row in rows}
    assert scores[("r1", "c3")] == 0.8
    assert scores[("r2", "c1")] == 0.0
    assert scores[("r2", "c3")] is None
    assert len(fake_llm) == 1

    again, _ = evaluate(PLAN, runs)
    assert again == []
    assert len(load_evals()) == 6


def test_caps_runs_per_version(fake_llm):
    runs = [run(f"r{index}") for index in range(10)] + [run("other", commit="bbb")]
    rows, _ = evaluate(PLAN, runs, per_version=3)
    assert {row["run_id"] for row in rows} == {"r9", "r8", "r7", "other"}


def test_model_errors_are_reported_not_stored(monkeypatch):
    from ai_lens import llm

    def unavailable(*args, **kwargs):
        raise llm.LensError("no key")

    monkeypatch.setattr(llm, "ask_json", unavailable)
    rows, errors = evaluate(PLAN, [run("r1")])
    assert errors == ["no key"]
    assert {row["check_id"] for row in rows} == {"c1", "c2"}


def test_video_checks_use_real_frames(fake_llm, video):
    result = EVALUATORS["video_integrity"].run({"output": str(video)}, {})
    assert result["score"] == 1.0
    assert len(media.frames(video)) == 4
    judged = EVALUATORS["judge"].run({"output": str(video), "output_type": "video", "inputs": {}}, {"check": "Smooth?"})
    assert judged["score"] == 0.8
    assert len(fake_llm[0]["images"]) == 4


def test_errors_from_the_users_model_are_reported(monkeypatch):
    from ai_lens.config import settings

    def broken_model(parts):
        raise TimeoutError("model timed out")

    monkeypatch.setattr(settings, "model", broken_model)
    _, errors = evaluate(PLAN, [run("r1")])
    assert errors == ["Your model raised TimeoutError: model timed out"]
