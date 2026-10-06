import json

import pytest

import ai_lens as lens
from ai_lens import references
from ai_lens.evaluators import EVALUATORS
from ai_lens.plan import describe, make_plan


def test_references_file_is_loaded_and_media_copied(tmp_path):
    folder = tmp_path / "evals"
    folder.mkdir()
    (folder / "best.png").write_bytes(b"best")
    lines = [
        {"output": "best.png", "inputs": {"prompt": "a cat"}, "notes": "sharp fur"},
        {"output": "A good caption names the subject."},
    ]
    (folder / "best.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    lens.configure(references=folder / "best.jsonl")
    image, text = references.load()
    assert image["kind"] == "image"
    assert image["output"]["artifact"].endswith(".png")
    assert text["kind"] == "text"


def test_matching_prefers_references_with_the_same_inputs():
    found = [
        {"inputs": {"prompt": "a cat"}, "output": "cat", "kind": "text"},
        {"inputs": {}, "output": "general", "kind": "text"},
    ]
    assert references.matching({"inputs": {"prompt": "a cat", "seed": 1}}, found)[0]["output"] == "cat"
    assert references.matching({"inputs": {"prompt": "a dog"}}, found)[0]["output"] == "general"


def test_missing_reference_file_only_warns():
    with pytest.warns(RuntimeWarning, match="reference file not found"):
        lens.configure(references=[{"output": "missing.mp4"}])
    assert references.load() == []


def test_track_builds_a_rubric_from_references(fake_llm):
    lens.configure(references=[{"output": "Short, vivid captions that name the subject."}])
    plan = make_plan("are captions getting better?", "text")
    check = plan["checks"][-1]
    assert check["evaluator"] == "reference"
    assert check["rubric"][0]["name"] == "Sharp subject"
    assert plan["references"] == references.fingerprint(references.load())
    assert describe(check) == "reference: matches your best results (1 criteria)"


def test_reference_judge_shows_reference_then_new_output(fake_llm, video):
    lens.configure(references=[{"output": str(video), "notes": "smooth"}])
    run = {"output": str(video), "output_type": "video", "inputs": {"prompt": "x"}, "prompt": "x"}
    result = EVALUATORS["reference"].run(run, {"rubric": [{"name": "Sharp subject", "description": "sharp"}]})
    assert result["score"] == 0.7
    assert result["criteria"] == {"Sharp subject": 0.6}
    parts = fake_llm[0]["parts"]
    labels = [part for part in parts if isinstance(part, str)]
    assert labels.index("Reference 1 (developer's notes: smooth):") < labels.index("The new output:")
    assert len(fake_llm[0]["images"]) == 8
