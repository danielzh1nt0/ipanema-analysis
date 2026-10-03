# Kaggle (free T4), 3 Oct (S3b): does pasting help the ball finder? The B7 ball-finder training, same settings and seed (no
# negatives, no at-feet copies: B7 is the app's finder and the baseline), with ONE far pasted real-ball patch (4-8 px, sized
# from the players in the frame, ipanema/pasteball.py) inside every training ball crop - frames whose real ball is labelled,
# so the finder is never taught to ignore a real ball. Graded by the script on the exam (B7 84/108) and the 34 clip moments;
# candidates over the SFK-BP test clip with the new and the old finder (same job) for tools/newfinder_grade.py (34-key, B4
# key, graded passes through the picker). Test clip and exam frames are never trained on and never give a patch.
import os, subprocess, runpy
os.environ["PASTE"] = "1"; os.environ["CANDS_CLIPS"] = "SFKBP1109_s1200"; os.environ["R2_BASE"] = "{{R2}}"
REPO = "/kaggle/temp/ia"; os.makedirs("/kaggle/temp", exist_ok=True)
subprocess.run(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True, check=True)
runpy.run_path(f"{REPO}/kaggle/ballfinder_rfdetr.py", run_name="__main__")
