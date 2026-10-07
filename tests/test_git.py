import shutil
import subprocess

import pytest

from ai_lens import git


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], check=True, cwd=tmp_path)
    (tmp_path / "prompt.py").write_text("PROMPT = 'one'\n")
    subprocess.run(["git", "add", "."], check=True, cwd=tmp_path)
    subprocess.run(["git", "commit", "-qm", "first"], check=True, cwd=tmp_path)
    return tmp_path


def test_each_set_of_edits_gets_its_own_fingerprint(repo):
    clean = git.git_info()
    assert clean.commit and clean.edits is None and clean.dirty is False
    (repo / "prompt.py").write_text("PROMPT = 'two'\n")
    first = git.git_info()
    (repo / "prompt.py").write_text("PROMPT = 'three'\n")
    second = git.git_info()
    assert first.edits and second.edits and first.edits != second.edits
    assert first.edited == ["prompt.py"]
    assert first.commit == clean.commit


def test_untracked_outputs_do_not_make_a_new_version(repo):
    (repo / "output.mp4").write_bytes(b"x")
    state = git.git_info()
    assert state.dirty is True
    assert state.edits is None


def test_outside_a_repo():
    assert git.git_info() == git.GitState()
