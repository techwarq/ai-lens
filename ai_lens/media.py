import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"},
}
OUTPUT_KINDS = ("text", "json", "image", "video", "audio")


def kind_of(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    return next((kind for kind, suffixes in KINDS.items() if suffix in suffixes), None)


def as_media_path(value: Any) -> Path | None:
    if isinstance(value, dict):
        value = value.get("artifact") or value.get("path")
    if not isinstance(value, (str, Path)) or len(str(value)) > 1024 or "\n" in str(value):
        return None
    path = Path(value)
    if kind_of(path) is None or not path.is_file():
        return None
    return path


def output_kind(value: Any) -> str:
    if isinstance(value, dict) and "sha256" in value:
        return kind_of(value["path"]) or "json"
    path = as_media_path(value)
    if path is not None:
        return kind_of(path) or "text"
    if value is None:
        return "none"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return "json"
    return "object"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess[bytes] | None:
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result if result.returncode == 0 else None


def probe(path: Path) -> dict[str, Any] | None:
    result = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    if result is None:
        return None
    return json.loads(result.stdout)


def stream(info: dict[str, Any], codec_type: str) -> dict[str, Any] | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == codec_type), None)


def duration(info: dict[str, Any]) -> float:
    try:
        return float(info.get("format", {}).get("duration") or 0)
    except ValueError:
        return 0.0


def _jpeg(path: Path, width: int, seek: float | None = None) -> bytes | None:
    args = ["ffmpeg", "-v", "error"]
    if seek is not None:
        args += ["-ss", f"{seek:.3f}"]
    args += ["-i", str(path), "-frames:v", "1", "-vf", f"scale=w=min(iw\\,{width}):h=-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    result = _run(args)
    return result.stdout if result and result.stdout else None


def image_jpeg(path: Path, width: int = 1024) -> bytes | None:
    return _jpeg(path, width)


def frames(path: Path, count: int = 4, width: int = 768) -> list[bytes]:
    info = probe(path)
    if info is None:
        return []
    length = duration(info)
    shots = [_jpeg(path, width, length * (index + 0.5) / count) for index in range(count)]
    return [shot for shot in shots if shot]


def pictures(path: Path) -> list[bytes]:
    if kind_of(path) == "video":
        return frames(path)
    jpeg = image_jpeg(path)
    return [jpeg] if jpeg else []


def silence_seconds(path: Path, threshold_db: int = -50, min_seconds: float = 0.5) -> float | None:
    args = ["ffmpeg", "-v", "info", "-i", str(path), "-af", f"silencedetect=n={threshold_db}dB:d={min_seconds}", "-f", "null", "-"]
    result = _run(args)
    if result is None:
        return None
    log = result.stderr.decode(errors="replace")
    return sum(float(match) for match in re.findall(r"silence_duration: ([\d.]+)", log))
