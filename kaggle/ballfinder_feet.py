# Kaggle (free T4), 2 Oct (S8, Daniel: "do all but not modal"): the B7 ball-finder training, same settings and seed, with balls at a
# player's feet / half hidden shown 3x more (AT_FEET=1: labelled balls in the lower half of a person box get 3 extra crops that
# keep the player). This is the finder's measured weak spot (ball at a foot 0.11 vs a line mark 0.19). Also the explicit
# not-a-ball crops from the previous run (no harm, no gain). Test clip untouched. Writes candidates over the SFK-BP clip
# with the new and the old finder for the offline grading (tools/newfinder_grade.py).
import os, subprocess, runpy
os.environ["AT_FEET"] = "1"; os.environ["FEET_COPIES"] = "3"
os.environ["NEG_JSON"] = "results/ball/negatives_2026-10-02.json"; os.environ["NEG_CLIPS"] = "p15u-vs-aik-2026-09-21-bd09_s2520"; os.environ["NEG_SFK_FP"] = "1"; os.environ["NEG_COPIES"] = "2"
os.environ["CANDS_CLIPS"] = "SFKBP1109_s1200"; os.environ["R2_BASE"] = "{{R2}}"
REPO = "/kaggle/temp/ia"; os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True, check=True)
runpy.run_path(f"{REPO}/kaggle/ballfinder_rfdetr.py", run_name="__main__")
