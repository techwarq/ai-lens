import functools
import subprocess


def _git(args: list[str], timeout: float = 2) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@functools.lru_cache(maxsize=1)
def git_info() -> tuple[str | None, str | None, bool | None]:
    commit = _git(["rev-parse", "HEAD"])
    branch = _git(["branch", "--show-current"])
    status = _git(["status", "--porcelain"])
    dirty = None if status is None else bool(status)
    return commit, branch, dirty


def message(commit: str) -> str | None:
    return _git(["log", "-1", "--format=%s", commit])


def diff(old: str, new: str | None = None, limit: int = 15_000) -> str:
    text = _git(["diff", "--stat", "--patch", old, *([new] if new else [])], timeout=10) or ""
    return text[:limit]
