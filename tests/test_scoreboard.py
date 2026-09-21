import subprocess, sys, os, tempfile, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_scoreboard_line():
    d = tempfile.mkdtemp(); os.makedirs(f"{d}/results"); log = f"{d}/r.txt"
    open(log, "w").write('--- SUMMARY ---\nx (ipanema 0.23.0\nball: WASB-only candidates, 2.9/frame\nball check (other picker, v1): 18/34 correct, ceiling 25/34\n'
                         'SUMMARY {"ball_check": {"correct": 24, "total": 34, "ceiling": 25}, "ball_grade": {"possession_ok": true}}')
    out = subprocess.run([sys.executable, f"{ROOT}/scripts_scoreboard.py", log, "abc1234", "TEST"], cwd=d, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    row = open(f"{d}/results/scoreboard.md").read().splitlines()[-1]
    assert "| 24/34 | 25 |" in row and "other picker v1: 18/34" in row and "| OK |" in row
