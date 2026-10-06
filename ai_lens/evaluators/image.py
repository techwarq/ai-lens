from typing import Any

from .. import media
from .basic import score


def integrity(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    path = media.as_media_path(run.get("output"))
    if path is None:
        return score(0.0, "image file not found")
    if not media.has_ffmpeg():
        return score(None, "ffmpeg is not installed")
    info = media.probe(path)
    picture = media.stream(info, "video") if info else None
    if picture is None:
        return score(0.0, "file does not decode as an image")
    width, height = picture.get("width") or 0, picture.get("height") or 0
    if width < 16 or height < 16:
        return score(0.0, f"suspicious size {width}x{height}")
    return score(1.0, f"{width}x{height}")
