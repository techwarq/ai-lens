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


def _render(path: Path, graph: str) -> bytes | None:
    args = ["ffmpeg", "-v", "error", "-i", str(path), "-lavfi", f"{graph},format=yuvj420p", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    result = _run(args)
    return result.stdout if result and result.stdout else None


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


def audio_pictures(path: Path) -> list[bytes]:
    shots = [_render(path, "showwavespic=s=1024x256:split_channels=0"), _render(path, "showspectrumpic=s=1024x384:legend=0")]
    return [shot for shot in shots if shot]


def pictures(path: Path) -> list[bytes]:
    if kind_of(path) == "video":
        return frames(path)
    if kind_of(path) == "audio":
        return audio_pictures(path)
    jpeg = image_jpeg(path)
    return [jpeg] if jpeg else []


def silence_seconds(path: Path, threshold_db: int = -50, min_seconds: float = 0.5) -> float | None:
    args = ["ffmpeg", "-v", "info", "-i", str(path), "-af", f"silencedetect=n={threshold_db}dB:d={min_seconds}", "-f", "null", "-"]
    result = _run(args)
    if result is None:
        return None
    log = result.stderr.decode(errors="replace")
    return sum(float(match) for match in re.findall(r"silence_duration: ([\d.]+)", log))


def volume(path: Path) -> tuple[float, float] | None:
    result = _run(["ffmpeg", "-v", "info", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    if result is None:
        return None
    log = result.stderr.decode(errors="replace")
    mean, peak = re.search(r"mean_volume: ([-\d.]+) dB", log), re.search(r"max_volume: ([-\d.]+) dB", log)
    return (float(mean.group(1)), float(peak.group(1))) if mean and peak else None


def audio_facts(path: Path) -> str:
    info = probe(path)
    if info is None:
        return "Could not measure the audio."
    facts = [f"{duration(info):.1f}s long"]
    sound = stream(info, "audio") or {}
    if sound.get("sample_rate"):
        facts.append(f"{sound['sample_rate']} Hz, {sound.get('channels', '?')} channel(s)")
    levels = volume(path)
    if levels:
        facts.append(f"average loudness {levels[0]:.0f} dB, peak {levels[1]:.1f} dB")
    silent = silence_seconds(path)
    if silent is not None:
        facts.append(f"{silent:.1f}s of silence")
    return "Measured: " + ", ".join(facts) + "."
