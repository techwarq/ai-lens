import shutil
import subprocess
from pathlib import Path

import pytest

from ai_lens.config import settings


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_LENS_DIR", str(tmp_path / ".ai-lens"))
    monkeypatch.delenv("AI_LENS_DISABLED", raising=False)
    monkeypatch.setattr(settings, "path", None)
    monkeypatch.setattr(settings, "enabled", True)
    monkeypatch.setattr(settings, "model", None)
    yield


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    def answer(parts, schema):
        calls.append({"parts": parts, "images": [part for part in parts if isinstance(part, bytes)]})
        return reply(schema), 0.001

    def reply(schema):
        keys = set(schema["properties"])
        if keys == {"score", "reason"}:
            return {"score": 0.8, "reason": "looks right"}
        if keys == {"notes", "score", "reason"}:
            return {"notes": ["asked for a caption", "caption names the subject"], "score": 7, "reason": "clear but plain"}
        if keys == {"steps"}:
            return {"steps": ["Check the subject is named.", "Check the mood comes through."]}
        if keys == {"checks"}:
            return {"checks": [{"evaluator": "judge", "check": "Is it on topic?"}]}
        if keys == {"criteria"}:
            return {"criteria": [{"name": "Sharp subject", "description": "The subject stays sharp."}]}
        if keys == {"summary"}:
            return {"summary": "The new prompt makes up numbers that are not on the site."}
        if keys == {"criteria", "score", "reason"}:
            return {"criteria": [{"name": "Sharp subject", "score": 0.6, "reason": "slightly soft"}], "score": 0.7, "reason": "softer than reference"}
        return {
            "summary": "One regression.",
            "suggestions": [
                {
                    "title": "Revert the model switch",
                    "impact": "quality",
                    "file": "app.py",
                    "line": 3,
                    "problem": "quality dropped",
                    "change": "use the old model",
                    "expected": "+0.2 quality",
                }
            ],
        }

    from ai_lens import llm

    monkeypatch.setattr(llm, "ask_json", answer)
    return calls


@pytest.fixture
def video(tmp_path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def audio(tmp_path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(path)], check=True)
    return path
