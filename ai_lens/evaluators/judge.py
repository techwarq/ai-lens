import json
from typing import Any

from .. import llm, media
from .basic import clamp, score
from .text import as_text

Parts = list[str | bytes]

SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}

INSTRUCTION = """Grade only the check above. Give a score from 0 to 1, where 1 fully meets the check and 0 fails it completely, and a one-sentence reason that names what you saw."""


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def show_inputs(run: dict[str, Any]) -> Parts:
    parts: Parts = []
    prompt = run.get("prompt")
    if prompt:
        parts.append(f"Prompt:\n{clip(prompt, 4_000)}")
    other = {key: value for key, value in (run.get("inputs") or {}).items() if value != prompt}
    if other:
        parts.append("Inputs:\n" + clip(json.dumps(other, default=str, indent=2), 3_000))
    for asset in run.get("assets") or []:
        path = media.as_media_path(asset)
        pictures = media.pictures(path) if path else []
        if pictures:
            parts += [f"Input asset '{asset['input']}':", *pictures]
    return parts or ["No inputs recorded."]


AUDIO = (
    "The audio, which you can't hear, shown as two pictures: a waveform (loudness over time, left to right) "
    "and a spectrogram (pitch over time, low notes at the bottom, brighter means louder). "
    "Judge only what the pictures and measurements show."
)


def show_output(value: Any, kind: str | None) -> Parts | None:
    if kind not in ("image", "video", "audio"):
        return [clip(as_text(value), 20_000)]
    path = media.as_media_path(value)
    pictures = media.pictures(path) if path else []
    if path is None or not pictures:
        return None
    if kind == "audio":
        return [AUDIO, *pictures, media.audio_facts(path)]
    label = f"{len(pictures)} frames sampled evenly across the video, in order:" if kind == "video" else "The image:"
    return [label, *pictures]


def unshowable(kind: str | None) -> dict[str, Any]:
    return score(None, f"could not read the {kind} output; is ffmpeg installed?")


def judge(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    kind = run.get("output_type")
    output = show_output(run.get("output"), kind)
    if output is None:
        return unshowable(kind)
    parts: Parts = [
        "You are grading one output of an AI application.",
        f"Check: {check['check']}",
        "What the application was given:",
        *show_inputs(run),
        "What it produced:",
        *output,
        INSTRUCTION,
    ]
    data, cost = llm.ask_json(parts, SCHEMA)
    return score(clamp(data["score"]), str(data["reason"]), cost)
