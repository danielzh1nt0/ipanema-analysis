"""tools/push_retry.sh: a bot commit that clashes with another bot's commit on the same file still gets pushed
(our side wins), and a push that cannot happen makes the script fail instead of passing silently."""
import os, subprocess, shutil, pytest
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "tools", "push_retry.sh")
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="no git")

def sh(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)

def clone(remote, path):
    sh(f"git clone -q {remote} {path}", ".")
    sh("git config user.name t && git config user.email t@t", path)

def test_conflict_is_resolved_and_pushed(tmp_path):
    remote, a, b = tmp_path / "r.git", tmp_path / "a", tmp_path / "b"
    sh(f"git init -q --bare -b main {remote}", ".")
    clone(remote, a); (a / "log.txt").write_text("start\n"); sh("git add . && git commit -qm init && git push -q origin HEAD:main", a)
    clone(remote, b)
    (a / "log.txt").write_text("start\nprogress from the tracking job\n"); sh("git commit -qam progress && git push -q origin HEAD:main", a)
    (b / "log.txt").write_text("start\nfinal from the scorecard\n"); (b / "card.md").write_text("card\n")
    sh("git add . && git commit -qm card", b)
    r = sh(f"bash {os.path.abspath(SCRIPT)} 2 0", b)
    assert r.returncode == 0, r.stdout + r.stderr
    sh("git pull -q origin main", a)
    assert (a / "card.md").exists() and "final from the scorecard" in (a / "log.txt").read_text()

def test_fails_loudly_when_push_impossible(tmp_path):
    a = tmp_path / "a"; sh(f"git init -q -b main {a}", "."); sh("git config user.name t && git config user.email t@t", a)
    (a / "x").write_text("x"); sh("git add . && git commit -qm x && git remote add origin /nonexistent/repo.git", a)
    r = sh(f"bash {os.path.abspath(SCRIPT)} 2 0", a)
    assert r.returncode == 1 and "PUSH FAILED" in r.stdout
