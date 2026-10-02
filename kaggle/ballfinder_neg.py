# Kaggle (free T4), 2 Oct (S8 "do it accurately"): the B7 ball-finder training, same settings and seed, plus explicit 'not a ball'
# crops: (1) spots mined from the AIK clip inputs (candidates away from the keyed ball, still spots with nobody near:
# results/ball/negatives_2026-10-02.json) and (2) the old finder's confident false guesses on SFK-BP TRAIN click frames
# (outside the exam and the test clip's window). The SFK-BP clip (34 moments, B4, graded passes) is never trained on.
# Also writes candidates over the SFK-BP clip with the new AND the old finder, for the offline picker/pass grading.
import os, subprocess, runpy
os.environ["NEG_JSON"] = "results/ball/negatives_2026-10-02.json"; os.environ["NEG_CLIPS"] = "p15u-vs-aik-2026-09-21-bd09_s2520"
os.environ["NEG_PER_CLIP"] = os.environ.get("NEG_PER_CLIP", "400"); os.environ["NEG_SFK_FP"] = "1"; os.environ["NEG_COPIES"] = "2"
os.environ["CANDS_CLIPS"] = "SFKBP1109_s1200"; os.environ["R2_BASE"] = "{{R2}}"
REPO = "/kaggle/temp/ia"; os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True, check=True)
runpy.run_path(f"{REPO}/kaggle/ballfinder_rfdetr.py", run_name="__main__")
