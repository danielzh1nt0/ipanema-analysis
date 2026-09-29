"""Run a script on Kaggle's free GPU from the free GitHub runner (28 Sep). No Modal.
    python tools/kaggle_run.py kaggle/<script>.py [max_minutes] [fetch]
fetch (29 Sep): don't push a new version, only wait for / download the output of the kernel already there
(a finished job whose results commit was lost).
Pushes a private GPU+internet script kernel, waits (log every 2 min), downloads its output to results/kaggle/<name>/.
'{{R2}}' in the script is replaced by the R2 public URL (a public bucket; the kernel is private)."""
import os, sys, json, time, subprocess, shutil, re
script = sys.argv[1]; max_min = float(sys.argv[2]) if len(sys.argv) > 2 else 300
name = os.path.splitext(os.path.basename(script))[0]; OUT = f"results/kaggle/{name}"; os.makedirs(OUT, exist_ok=True)
def sh(c, check=False):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); txt = (r.stdout + r.stderr).strip()
    if check and r.returncode: print(txt); sys.exit(f"failed: {c}")
    return txt
def user():
    u = os.environ.get("KAGGLE_USERNAME")
    if u: return u
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        a = KaggleApi(); a.authenticate()
        for k in ("username",):
            v = a.get_config_value(k) if hasattr(a, "get_config_value") else None
            if v: return v
        v = getattr(a, "config_values", {}).get("username")
        if v: return v
    except Exception as e: print("api user lookup:", repr(e)[:300])
    m = re.search(r"username:\s*(\S+)", sh("kaggle config view"))
    if m and m.group(1) != "None": return m.group(1)
    sys.exit("could not find the Kaggle username: set KAGGLE_USERNAME")
u = user(); slug = f"ipanema-{name}".replace("_", "-").lower(); kid = f"{u}/{slug}"; print("kernel", kid, flush=True)
d = f"/tmp/kaggle_{name}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
code = open(script).read().replace("{{R2}}", os.environ.get("R2_PUBLIC_URL", "").rstrip("/"))
open(f"{d}/{name}.py", "w").write(code)
json.dump({"id": kid, "title": slug, "code_file": f"{name}.py", "language": "python", "kernel_type": "script", "is_private": True,
           "enable_gpu": True, "enable_internet": True, "machine_shape": os.environ.get("KAGGLE_GPU", "NvidiaTeslaT4"), "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open(f"{d}/kernel-metadata.json", "w"), indent=1)
if "fetch" in sys.argv[3:]: print("fetch only: no new version pushed", flush=True)
else:
    # 29 Sep: Kaggle allows 2 GPU batch sessions at once; a refused push used to look like a finished job. Wait for a slot.
    tw = time.time()
    while True:
        out = sh(f"kaggle kernels push -p {d}", check=True); print(out, flush=True)
        if "push error" not in out.lower(): break
        if "session count" not in out.lower() or time.time() - tw > float(os.environ.get("KAGGLE_SLOT_WAIT_MIN", "120")) * 60:
            json.dump({"kernel": kid, "status": "push refused: " + out[-300:], "minutes": 0}, open(f"{OUT}/run.json", "w"), indent=1); sys.exit("push refused")
        print(f"{(time.time() - tw) / 60:5.1f} min: both Kaggle GPU slots busy, retrying in 3 min", flush=True); time.sleep(180)
t0 = time.time(); last = ""
while True:
    time.sleep(30); st = sh(f"kaggle kernels status {kid}"); low = st.lower()
    if st != last or int(time.time() - t0) % 120 < 30: print(f"{(time.time() - t0) / 60:5.1f} min: {st[-200:]}", flush=True); last = st
    if "404" in low and time.time() - t0 < 600: continue                      # a just-pushed kernel can 404 for a while
    if any(w in low for w in ("complete", "error", "cancel")): break
    if time.time() - t0 > max_min * 60: print("gave up waiting (the kernel may still be running on Kaggle)"); break
print(sh(f"kaggle kernels output {kid} -p {OUT}"), flush=True)
json.dump({"kernel": kid, "status": st, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/run.json", "w"), indent=1)
for f in sorted(os.listdir(OUT)): print(" ", f, os.path.getsize(f"{OUT}/{f}"))
