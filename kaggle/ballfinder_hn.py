# Kaggle (free T4), 1 Oct (Daniel: "fix ball"): the B7 ball-finder training run, unchanged settings (seed 0), plus hard-example
# oversampling (HARD_NEG=1): training crops where the current finder is confidently wrong (a shoe, a head, a second ball) are
# shown 3x. Graded on the same SFK-BP exam (108) + 34 clip moments as B7 (exam 84/108, top guess 70/81; clip top guess 26/34).
import os, subprocess, runpy
os.environ["HARD_NEG"] = "1"; os.environ["R2_BASE"] = "{{R2}}"
REPO = "/kaggle/temp/ia"; os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True, check=True)
runpy.run_path(f"{REPO}/kaggle/ballfinder_rfdetr.py", run_name="__main__")
