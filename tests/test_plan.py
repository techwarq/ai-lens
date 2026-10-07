import pytest

from ai_lens import llm
from ai_lens.plan import make_plan
from ai_lens.store import load_plan


def test_model_plan_is_cleaned_and_saved(fake_llm):
    plan = make_plan("are my answers on topic?", "text")
    assert [check["evaluator"] for check in plan["checks"]] == ["success", "judge"]
    assert plan["checks"][1]["check"] == "Is it on topic?"
    assert plan["planner"] == "model"
    assert load_plan() == plan


def test_unknown_and_duplicate_checks_are_dropped(monkeypatch):
    checks = [
        {"evaluator": "judge", "check": "Smooth motion?"},
        {"evaluator": "judge", "check": "Smooth motion?"},
        {"evaluator": "made_up", "check": ""},
        {"evaluator": "judge", "check": ""},
        {"evaluator": "video_integrity", "check": "ignored"},
    ]
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: ({"checks": checks}, 0.0))
    plan = make_plan("videos", "video")
    assert [(check["evaluator"], check["check"]) for check in plan["checks"]] == [
        ("success", ""),
        ("judge", "Smooth motion?"),
        ("video_integrity", ""),
    ]


def test_falls_back_to_default_checks_without_a_model(monkeypatch):
    def unavailable(*args, **kwargs):
        raise llm.LensError("no key")

    monkeypatch.setattr(llm, "ask_json", unavailable)
    plan = make_plan("videos", "video")
    assert plan["planner"] == "default"
    assert {check["evaluator"] for check in plan["checks"]} == {"success", "geval", "video_integrity"}


@pytest.mark.parametrize("kind", ["text", "json", "image", "video", "audio"])
def test_every_output_type_gets_a_plan(monkeypatch, kind):
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: (_ for _ in ()).throw(llm.LensError("x")))
    plan = make_plan("goal", kind)
    assert plan["kind"] == kind
    assert len(plan["checks"]) >= 2


def test_plain_text_defaults_do_not_demand_json(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: (_ for _ in ()).throw(llm.LensError("x")))
    plan = make_plan("answers", "text")
    assert [check["evaluator"] for check in plan["checks"]] == ["success", "not_empty", "geval"]


def test_geval_checks_get_evaluation_steps(monkeypatch, fake_llm):
    replies = {"checks": [{"evaluator": "geval", "check": "Coherence: each sentence follows from the last."}]}
    original = llm.ask_json
    monkeypatch.setattr(llm, "ask_json", lambda parts, schema: (replies, 0.0) if "checks" in schema["properties"] else original(parts, schema))
    plan = make_plan("are captions coherent?", "text")
    assert plan["checks"][1]["steps"] == ["Check the subject is named.", "Check the mood comes through."]


def test_tool_use_is_checked_when_the_app_calls_tools(monkeypatch):
    from ai_lens import plan as planning

    runs = [{"id": "r1", "output": "hi", "output_type": "text", "steps": [{"id": "s1", "name": "search", "type": "tool"}]}]
    monkeypatch.setattr(planning, "load_runs", lambda: runs)
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: (_ for _ in ()).throw(llm.LensError("x")))
    plan = make_plan("answers", "text")
    trajectory = plan["checks"][-1]
    assert trajectory["evaluator"] == "trajectory"
    assert len(trajectory["steps"]) >= 3


def test_trajectory_is_not_offered_without_steps(monkeypatch):
    monkeypatch.setattr(llm, "ask_json", lambda *args, **kwargs: ({"checks": [{"evaluator": "trajectory", "check": "Tools ok?"}]}, 0.0))
    plan = make_plan("answers", "text")
    assert [check["evaluator"] for check in plan["checks"]] == ["success"]
