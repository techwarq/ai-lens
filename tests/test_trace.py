import asyncio
import threading
from pathlib import Path

import pytest

import ai_lens as lens
from ai_lens import store
from ai_lens.store import load_runs

RESPONSE = {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 1000, "output_tokens": 200}}
RESPONSE_COST = (1000 * 2 + 200 * 10) / 1_000_000


def test_records_inputs_output_and_timing():
    @lens.trace
    def make_video(prompt, seed=42):
        return f"video of {prompt}"

    assert make_video("a cat") == "video of a cat"
    [run] = load_runs()
    assert run["name"] == "make_video"
    assert run["inputs"] == {"prompt": "a cat", "seed": 42}
    assert run["output"] == "video of a cat"
    assert run["output_type"] == "text"
    assert run["status"] == "ok"
    assert run["latency_s"] >= 0
    assert "test_trace.py:" in run["source"]


def test_error_is_recorded_and_reraised():
    @lens.trace
    def broken():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        broken()
    [run] = load_runs()
    assert run["status"] == "error"
    assert run["error"]["type"] == "ValueError"
    assert "bad input" in run["error"]["traceback"]


def test_async_functions_record_the_awaited_output():
    @lens.trace
    async def generate(prompt):
        await asyncio.sleep(0)
        return prompt.upper()

    assert asyncio.run(generate("hi")) == "HI"
    assert load_runs()[0]["output"] == "HI"


def test_steps_nest_and_tokens_roll_up():
    @lens.step(type="tool")
    def search(query):
        return ["clip-1"]

    @lens.step
    def write_script(query):
        return RESPONSE

    @lens.trace(tags=["v2"])
    def app(query):
        search(query)
        with lens.step("plan"):
            write_script(query)
        return "done"

    app("cats")
    [run] = load_runs()
    names = {step["name"]: step for step in run["steps"]}
    assert set(names) == {"search", "plan", "write_script"}
    assert names["search"]["type"] == "tool"
    assert names["write_script"]["type"] == "llm"
    assert names["write_script"]["parent_id"] == names["plan"]["id"]
    assert run["tags"] == ["v2"]
    assert run["model"] == "claude-sonnet-5-5"
    assert run["input_tokens"] == 1000
    assert run["output_tokens"] == 200
    assert run["cost"] == pytest.approx(RESPONSE_COST)


def test_same_response_is_not_counted_twice():
    @lens.step
    def call_model():
        return RESPONSE

    @lens.trace
    def app():
        return call_model()

    app()
    assert load_runs()[0]["input_tokens"] == 1000


def test_context_manager_and_log():
    with lens.trace("batch") as run:
        lens.log(params={"seed": 7}, model="gen-v3", note="first try")
        run.output("result")

    [saved] = load_runs()
    assert saved["name"] == "batch"
    assert saved["params"] == {"seed": 7}
    assert saved["model"] == "gen-v3"
    assert saved["metadata"] == {"note": "first try"}
    assert saved["output"] == "result"


def test_step_outside_a_run_just_runs():
    @lens.step
    def helper(x):
        return x * 2

    assert helper(2) == 4
    assert load_runs() == []


def test_disabled_records_nothing(monkeypatch):
    monkeypatch.setenv("AI_LENS_DISABLED", "1")

    @lens.trace
    def app():
        return 1

    assert app() == 1
    assert load_runs() == []


def test_media_output_is_copied_so_overwrites_do_not_matter(tmp_path):
    path = tmp_path / "out.png"
    path.write_bytes(b"first")

    @lens.trace
    def render():
        return str(path)

    render()
    path.write_bytes(b"second")
    [run] = load_runs()
    assert run["output_type"] == "image"
    assert len(run["output"]["sha256"]) == 64
    assert Path(run["output"]["artifact"]).read_bytes() == b"first"


def test_save_failure_only_warns(monkeypatch):
    def fail(run):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_run", fail)

    @lens.trace
    def app():
        return "still works"

    with pytest.warns(RuntimeWarning, match="disk full"):
        assert app() == "still works"


def test_unserializable_values_are_stored_as_text():
    @lens.trace
    def app(lock):
        return {"lock": lock, "raw": b"\x00\x01"}

    app(threading.Lock())
    [run] = load_runs()
    assert "lock" in run["inputs"]["lock"]
    assert run["output"]["raw"] == "<2 bytes>"


def test_threads_write_whole_lines():
    @lens.trace
    def app(index):
        return index

    threads = [threading.Thread(target=lambda: [app(i) for i in range(25)]) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_runs()) == 200


def test_prompt_and_input_assets_are_tracked(tmp_path):
    image = tmp_path / "start.png"
    image.write_bytes(b"frame")
    style = tmp_path / "style.jpg"
    style.write_bytes(b"style")

    @lens.trace
    def animate(prompt, start_frame):
        lens.log(prompt=f"cinematic, {prompt}", assets=[str(style), str(image)])
        return "ok"

    animate("a cat surfing", str(image))
    image.write_bytes(b"changed later")
    [run] = load_runs()
    assert run["prompt"] == "cinematic, a cat surfing"
    assert [asset["input"] for asset in run["assets"]] == ["start_frame", "logged"]
    assert Path(run["assets"][0]["artifact"]).read_bytes() == b"frame"


def test_prompt_argument_is_picked_up():
    @lens.trace
    def generate(prompt):
        return prompt

    generate("a forest")
    assert load_runs()[0]["prompt"] == "a forest"
