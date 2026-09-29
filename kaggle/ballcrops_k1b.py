# Kaggle (free GPU): K1b (ball followed by picture matching): ball / not-ball crops at Daniel's checked balls on Vasalund, Solheim, Spånga, Djursholm -> /kaggle/working/*
import subprocess, os, glob, shutil, json, time
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
t0 = time.time()
sh("nvidia-smi --query-gpu=name --format=csv; df -h /kaggle/temp /kaggle/working | tail -2")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git /kaggle/temp/WASB-SBDT")
sh("pip install -q gdown omegaconf pyyaml")
env = dict(os.environ, R2_PUBLIC_URL="{{R2}}", WASB_DIR="/kaggle/temp/WASB-SBDT", WROOT="/kaggle/temp/wroot", K1_TMP="/kaggle/temp", K1_WINDOW="12", K1_HOW="template", K1_NEG="4")
r = subprocess.run(["python", "tools/k1_crops.py", "/kaggle/working"], cwd="/kaggle/temp/ia", env=env, capture_output=True, text=True)
open("/kaggle/working/log.txt", "w").write((r.stdout + r.stderr)[-20000:])
json.dump({"returncode": r.returncode, "minutes": round((time.time() - t0) / 60, 1)}, open("/kaggle/working/run_info.json", "w"))
print((r.stdout + r.stderr)[-3000:]); print("done", r.returncode, round((time.time() - t0) / 60, 1), "min")
