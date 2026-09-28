# Kaggle (free GPU): player tracking with the fixed kits on Reymersholm, 5 min from 1500 s -> /kaggle/working/*
import subprocess, os, glob, shutil, json, time
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
t0 = time.time()
sh("nvidia-smi --query-gpu=name --format=csv; python -c 'import torch;print(torch.__version__, torch.cuda.is_available())'")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("pip install -q rfdetr==1.11.0 supervision")
sh("python -c 'import torch;print(\"after install\", torch.__version__, torch.cuda.is_available())'")
M = "p15u-vs-reymersholm-2026-09-18"
env = dict(os.environ, MATCH=M, START_S="1500", DUR_S="300", SKIP_INSTALL="1", SAVE_ALL_ROWS="1", DETECTORS="rfdetr",
           IPANEMA_DET_BATCH="8", IPANEMA_LOG_EVERY="500", R2_PUBLIC_URL="{{R2}}")
r = subprocess.run(["python", "tools/tracktest.py"], cwd="/kaggle/temp/ia", env=env, capture_output=True, text=True)
open("/kaggle/working/log.txt", "w").write((r.stdout + r.stderr)[-20000:])
for f in glob.glob(f"/kaggle/temp/ia/results/qa/tracktest_{M}/*"): shutil.copy(f, "/kaggle/working/")
json.dump({"returncode": r.returncode, "minutes": round((time.time() - t0) / 60, 1)}, open("/kaggle/working/run_info.json", "w"))
print("done", r.returncode, round((time.time() - t0) / 60, 1), "min")
