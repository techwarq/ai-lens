from typing import Any

from .. import media
from .basic import score


def integrity(run: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    path = media.as_media_path(run.get("output"))
    if path is None:
        return score(0.0, "video file not found")
    if not media.has_ffmpeg():
        return score(None, "ffmpeg is not installed")
    info = media.probe(path)
    picture = media.stream(info, "video") if info else None
    if info is None or picture is None:
        return score(0.0, "file does not decode as a video")
    length = media.duration(info)
    if length <= 0:
        return score(0.0, "video has zero duration")
    return score(1.0, f"{picture.get('width')}x{picture.get('height')}, {length:.1f}s, {picture.get('avg_frame_rate')} fps")
