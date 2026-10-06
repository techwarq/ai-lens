from typing import Any

from .. import media
from .basic import score


def integrity(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    path = media.as_media_path(run.get("output"))
    if path is None:
        return score(0.0, "audio file not found")
    if not media.has_ffmpeg():
        return score(None, "ffmpeg is not installed")
    info = media.probe(path)
    if info is None or media.stream(info, "audio") is None:
        return score(0.0, "file does not decode as audio")
    length = media.duration(info)
    if length <= 0:
        return score(0.0, "audio has zero duration")
    return score(1.0, f"{length:.1f}s")


def silence(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    path = media.as_media_path(run.get("output"))
    if path is None:
        return score(None, "audio file not found")
    info = media.probe(path) if media.has_ffmpeg() else None
    silent = media.silence_seconds(path) if info else None
    length = media.duration(info) if info else 0.0
    if silent is None or length <= 0:
        return score(None, "could not analyse audio")
    ratio = min(1.0, silent / length)
    return score(1.0 - ratio, f"{silent:.1f}s of {length:.1f}s is silence")
