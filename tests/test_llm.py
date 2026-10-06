import json
from types import SimpleNamespace

import pytest

import ai_lens as lens
from ai_lens import llm
from ai_lens.config import settings

SCHEMA = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}


def test_without_a_model_nothing_is_called():
    with pytest.raises(llm.LensError, match="No model configured"):
        llm.ask_json(["hi"], SCHEMA)


def test_gets_text_and_images_in_order_and_parses_fenced_json():
    seen = []

    def model(parts):
        seen.append(parts)
        return 'Sure!\n```json\n{"score": 0.9}\n```'

    settings.model = model
    data, cost = llm.ask_json(["look at this", b"\xff\xd8jpeg"], SCHEMA)
    assert data == {"score": 0.9}
    assert cost == 0.0
    assert seen[0][:2] == ["look at this", b"\xff\xd8jpeg"]
    assert "JSON schema" in seen[0][-1]


def test_retries_once_when_the_reply_is_not_json():
    replies = iter(["I think it's good", '{"score": 1}'])
    settings.model = lambda parts: next(replies)
    assert llm.ask_json(["x"], SCHEMA)[0] == {"score": 1}


def test_gives_up_after_two_bad_replies():
    settings.model = lambda parts: "no idea"
    with pytest.raises(llm.LensError, match="valid JSON"):
        llm.ask_json(["x"], SCHEMA)


def test_provider_responses_are_read_and_priced():
    response = SimpleNamespace(
        model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="text", text='{"score": 0.5}')],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
    )
    settings.model = lambda parts: response
    data, cost = llm.ask_json(["x"], SCHEMA)
    assert data == {"score": 0.5}
    assert cost == pytest.approx((1000 * 2 + 100 * 10) / 1_000_000)


def test_model_location_is_saved_for_the_lens_command(tmp_path):
    (tmp_path / "my_judge.py").write_text("def judge(parts):\n    return '{\"score\": 0.4}'\n")
    lens.configure(model="my_judge:judge")
    assert json.loads((tmp_path / ".ai-lens" / "config.json").read_text()) == {"model": "my_judge:judge"}
    assert llm.ask_json(["x"], SCHEMA)[0] == {"score": 0.4}


def test_functions_the_command_cannot_import_only_warn():
    def local_model(parts):
        return '{"score": 1}'

    with pytest.warns(RuntimeWarning, match="importable module"):
        lens.configure(model=local_model)
    assert llm.ask_json(["x"], SCHEMA)[0] == {"score": 1}
