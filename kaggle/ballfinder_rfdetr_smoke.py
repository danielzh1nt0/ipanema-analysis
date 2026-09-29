# Kaggle smoke run (29 Sep) for kaggle/ballfinder_rfdetr.py: 50 training crops, 1 epoch, 3 exam frames + 3 clip moments.
# Proves install, R2 download, RF-DETR training and tiled inference before the real run.
import os, subprocess, runpy
os.environ["MODE"] = "smoke"; os.environ["R2_BASE"] = "{{R2}}"
subprocess.run("rm -rf /kaggle/temp/ia0 && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia0", shell=True)
runpy.run_path("/kaggle/temp/ia0/kaggle/ballfinder_rfdetr.py", run_name="__main__")
