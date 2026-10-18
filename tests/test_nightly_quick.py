"""N0-18c: the nightly test count must see pytest failures (a pytest-style file run as a plain script exits 0)."""
import os, subprocess, sys
sys.path.insert(0, "tools")
import nightly_quick as NQ

FAILING = "def test_x():\n    assert 1 == 2\n"
PASSING = "def test_x():\n    assert 1 == 1\n"
SCRIPT_OK = "def check():\n    assert 1 == 1\n\nif __name__ == '__main__':\n    check()\n"
SCRIPT_BAD = "def check():\n    assert 1 == 2\n\nif __name__ == '__main__':\n    check()\n"
EMPTY = "x = 1\n"


def _write(tmp_path, name, body):
    p = tmp_path / name; p.write_text(body); return str(p)


def test_real_pytest_runs(tmp_path):
    files = [_write(tmp_path, "test_fail.py", FAILING), _write(tmp_path, "test_pass.py", PASSING), _write(tmp_path, "test_empty.py", EMPTY),
             _write(tmp_path, "test_script_ok.py", SCRIPT_OK), _write(tmp_path, "test_script_bad.py", SCRIPT_BAD)]
    # pytest finds no test_ function in the script-style files (exit 5) -> they count as passing, like before
    assert NQ.failed_test_files(files, env=dict(os.environ)) == ["test_fail.py"]


def test_pytest_failure_not_hidden_by_script_run(tmp_path):
    """the old loop re-ran a failed file as a script, which exits 0 for a pytest-style file -> the failure vanished"""
    f = _write(tmp_path, "test_fail.py", FAILING)
    assert subprocess.run([sys.executable, f]).returncode == 0          # the trap
    assert NQ.failed_test_files([f], env=dict(os.environ)) == ["test_fail.py"]


class _R:
    def __init__(self, rc): self.returncode = rc


def test_script_style_second_chance_only_with_main(tmp_path):
    calls = []
    def fake(cmd, **kw):
        calls.append(cmd); return _R(1 if "-m" in cmd else 0)      # pytest fails, script passes
    with_main = _write(tmp_path, "test_a.py", SCRIPT_OK); without = _write(tmp_path, "test_b.py", FAILING)
    assert NQ.failed_test_files([with_main, without], run=fake) == ["test_b.py"]
    assert sum(1 for c in calls if "-m" not in c) == 1                 # only the __main__ file was re-run as a script


def test_script_style_failing_both_ways_is_failed(tmp_path):
    f = _write(tmp_path, "test_a.py", SCRIPT_BAD)
    assert NQ.failed_test_files([f], run=lambda cmd, **kw: _R(1)) == ["test_a.py"]


def test_importing_does_not_run_the_nightly():
    """importing only defines functions (before 18 Oct the whole nightly ran at import)"""
    assert callable(NQ.main) and callable(NQ.failed_test_files)
