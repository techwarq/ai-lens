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


def test_geval_follows_the_plan_steps_and_scores_out_of_ten(fake_llm):
    check = {"check": "Coherence", "steps": ["Read it.", "Check the flow."]}
    result = EVALUATORS["geval"].run(run("r1"), check)
    assert result["score"] == 0.7
    assert result["notes"] == ["asked for a caption", "caption names the subject"]
    assert "Evaluation steps:\n1. Read it.\n2. Check the flow." in fake_llm[0]["parts"]


def test_geval_hears_audio_as_pictures(fake_llm, audio):
    result = EVALUATORS["geval"].run({"output": str(audio), "output_type": "audio", "inputs": {}}, {"check": "Clean tone?"})
    assert result["score"] == 0.7
    assert len(fake_llm[0]["images"]) == 2
    assert any(isinstance(part, str) and part.startswith("Measured: 2.0s long") for part in fake_llm[0]["parts"])


def test_trajectory_shows_every_tool_call(fake_llm):
    steps = [
        {"id": "s1", "name": "plan", "type": "step", "inputs": {"q": "weather"}, "output": "use search"},
        {"id": "s2", "name": "search", "type": "tool", "parent_id": "s1", "inputs": {"city": "Pune"}, "output": {"temp": 31}},
        {"id": "s3", "name": "book", "type": "tool", "status": "error", "inputs": {}, "error": {"type": "KeyError", "message": "date"}},
    ]
    result = EVALUATORS["trajectory"].run({**run("r1"), "steps": steps}, {"check": "Right tools?"})
    assert result["score"] == 0.7
    shown = next(part for part in fake_llm[0]["parts"] if isinstance(part, str) and part.startswith("Every step"))
    assert '  2. [tool] search(city="Pune") returned {"temp": 31}' in shown
    assert "3. [tool] book() raised KeyError: date" in shown


def test_trajectory_needs_recorded_steps(fake_llm):
    result = EVALUATORS["trajectory"].run(run("r1"), {"check": "Right tools?"})
    assert result["score"] is None
    assert fake_llm == []
