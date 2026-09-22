"""The release log gets one entry per pipeline version, newest first, and never duplicates a version."""
import os, subprocess, sys, tempfile, shutil, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _run(d):
    return subprocess.run([sys.executable, f"{ROOT}/scripts_release.py"], cwd=d, capture_output=True, text=True)

def test_adds_one_entry_per_version():
    d = tempfile.mkdtemp(); os.makedirs(f"{d}/ipanema"); os.makedirs(f"{d}/results")
    subprocess.run(["git", "init", "-q", d], check=True)
    subprocess.run(["git", "-C", d, "commit", "-q", "--allow-empty", "-m", "first change"], check=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    open(f"{d}/ipanema/__init__.py", "w").write('__version__ = "1.2.3"\n')
    open(f"{d}/results/scoreboard.md", "w").write("| 2026-09-22 | abc | M | 1.2.3 | 24/34 | 25 | - | OK | withheld | 5 | notes |\n")
    assert _run(d).returncode == 0
    text = open(f"{d}/RELEASES.md").read()
    assert "## v1.2.3" in text and "first change" in text and "24/34" in text
    out = _run(d); assert "already in RELEASES.md" in out.stdout and open(f"{d}/RELEASES.md").read() == text      # no duplicate
    open(f"{d}/ipanema/__init__.py", "w").write('__version__ = "1.2.4"\n')
    subprocess.run(["git", "-C", d, "commit", "-q", "--allow-empty", "-m", "second change"], check=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
    _run(d); text2 = open(f"{d}/RELEASES.md").read()
    assert text2.index("## v1.2.4") < text2.index("## v1.2.3") and "second change" in text2                        # newest first
