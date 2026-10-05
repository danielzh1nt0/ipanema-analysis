# Kaggle (free GPU): K3c re-label of every exported Vallentuna player (tools/vall_relabel.py) -> /kaggle/working/*
import subprocess, os, glob, shutil, json, time
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
t0 = time.time(); M = "p15u-vs-vallentuna-2026-10-03-6cce"
sh("nvidia-smi --query-gpu=name --format=csv")
sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
sh("pip install -q rfdetr==1.11.0 supervision")
sh("wget -q -O /kaggle/temp/v.mp4 " + "{{R2}}" + "/" + M + "/video.mp4; ls -la /kaggle/temp/v.mp4")
env = dict(os.environ, VIDEO="/kaggle/temp/v.mp4", OUT="/kaggle/working/relabel", PYTHONPATH="/kaggle/temp/ia", STEP="2", BATCH="8")
r = subprocess.Popen(["python", "tools/vall_relabel.py"], cwd="/kaggle/temp/ia", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log = []
for line in r.stdout: print(line, end="", flush=True); log.append(line)
r.wait(); open("/kaggle/working/log.txt", "w").write("".join(log[-400:]))
json.dump({"returncode": r.returncode, "minutes": round((time.time() - t0) / 60, 1)}, open("/kaggle/working/run_info.json", "w"))
print("done", r.returncode, round((time.time() - t0) / 60, 1), "min")
