# Kaggle (free T4), H1b 1 Oct (Daniel: yes, testing only): the B7 ball-finder training run, unchanged settings (seed 0), plus the
# outside Bundesliga TV ball set (HF_EXTRA=1). Graded on the same SFK-BP exam (108) + 34 clip moments as B7 (exam 84/108,
# clip top guess 26/34). Weights are named *_hf_TESTONLY: the set's licence is doubtful, never promote them.
import os, subprocess, runpy
os.environ["HF_EXTRA"] = "1"; os.environ["R2_BASE"] = "{{R2}}"
REPO = "/kaggle/temp/ia"; os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True, check=True)
runpy.run_path(f"{REPO}/kaggle/ballfinder_rfdetr.py", run_name="__main__")
