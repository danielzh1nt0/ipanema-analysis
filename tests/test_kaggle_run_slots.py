"""kaggle_run.py waits for a free GPU slot when Kaggle refuses the push (2-session limit), instead of reporting a finished job."""
import os, subprocess, sys, stat

def test_push_retries_on_session_limit(tmp_path):
    bin_ = tmp_path / "bin"; bin_.mkdir(); state = tmp_path / "n"
    stub = bin_ / "kaggle"
    stub.write_text(f"""#!/bin/sh
case "$2" in
  push) n=$(cat {state} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {state}
        if [ $n -lt 2 ]; then echo "Kernel push error: Maximum batch GPU session count of 2 reached."; else echo "Kernel version 1 successfully pushed."; fi ;;
  status) echo 'has status "KernelWorkerStatus.COMPLETE"' ;;
  output) echo "Output file downloaded" ;;
esac
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    (tmp_path / "kaggle").mkdir(); (tmp_path / "kaggle" / "slottest.py").write_text("print('hi')\n")
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", KAGGLE_USERNAME="u")
    code = open(f"{repo}/tools/kaggle_run.py").read().replace("time.sleep(180)", "time.sleep(0)").replace("time.sleep(30)", "time.sleep(0)")
    (tmp_path / "kr.py").write_text(code)
    r = subprocess.run([sys.executable, str(tmp_path / "kr.py"), "kaggle/slottest.py", "1"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert state.read_text().strip() == "2"
    assert "slots busy" in r.stdout and "COMPLETE" in (tmp_path / "results/kaggle/slottest/run.json").read_text()
