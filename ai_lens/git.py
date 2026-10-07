import hashlib
import subprocess
from typing import NamedTuple


class GitState(NamedTuple):
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    edits: str | None = None
    edited: list[str] = []


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


def _edited(diff: str) -> list[str]:
    return [line.rpartition(" b/")[2] for line in diff.splitlines() if line.startswith("diff --git a/")]


def git_info() -> GitState:
    status = _git(["status", "--porcelain=v2", "--branch"])
    if status is None:
        return GitState()
    lines = status.splitlines()
    headers = {key: value for line in lines if line.startswith("# ") for key, _, value in [line[2:].partition(" ")]}
    commit = headers.get("branch.oid")
    commit = None if commit in (None, "(initial)") else commit
    branch = headers.get("branch.head")
    branch = None if branch in (None, "(detached)") else branch
    dirty = any(not line.startswith("#") for line in lines)
    diff = _git(["diff", "HEAD", "--no-ext-diff"], timeout=5) if commit and dirty else None
    if not diff:
        return GitState(commit, branch, dirty)
    return GitState(commit, branch, dirty, hashlib.sha256(diff.encode()).hexdigest()[:12], _edited(diff))


def message(commit: str) -> str | None:
    return _git(["log", "-1", "--format=%s", commit])


def diff(old: str, new: str | None = None, limit: int = 15_000) -> str:
    text = _git(["diff", "--stat", "--patch", old, *([new] if new else [])], timeout=10) or ""
    return text[:limit]
