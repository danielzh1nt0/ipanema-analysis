# Kaggle (free GPU, 29 Sep, E3 re-check): new RF-DETR ball finder guesses on the Reymersholm 5-min piece (from 1500 s,
# frames 44955 + 0..8991, every STRIDE-th frame) for tools/reymlab.py (who-has-the-ball, 32-moment key).
# Output: /kaggle/working/ball_cands_rfdetr_reym.json.gz = [[seg_frame, x, y, conf], ...] (top 10 per frame).
import os, sys, json, gzip, time, subprocess, traceback
R2 = os.environ.get("R2_BASE", "{{R2}}").rstrip("/"); WORK = os.environ.get("WORK", "/kaggle/working"); TMP = os.environ.get("TMPD", "/kaggle/temp")
os.makedirs(WORK, exist_ok=True); LOCAL = os.environ.get("LOCAL") == "1"
M = "p15u-vs-reymersholm-2026-09-18"; OFFSET = 44955; N = int(os.environ.get("N", "30" if LOCAL else "8992")); STRIDE = int(os.environ.get("STRIDE", "2"))
t0 = time.time(); LOG = []; REP = {"started": time.strftime("%Y-%m-%d %H:%M"), "stride": STRIDE}
def log(m): LOG.append(f"{(time.time() - t0) / 60:5.1f} min  {m}"); print(LOG[-1], flush=True); open(f"{WORK}/log.txt", "w").write("\n".join(LOG) + "\n")
def save(): REP["minutes"] = round((time.time() - t0) / 60, 1); json.dump(REP, open(f"{WORK}/result.json", "w"), indent=1)
try:
    REPO = os.environ.get("REPO_DIR", f"{TMP}/ia")
    if not LOCAL:
        subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}", shell=True)
        subprocess.run('pip install -q "rfdetr==1.11.0"', shell=True)
    sys.path.insert(0, REPO)
    import cv2
    from ipanema import ballrf as BR
    model = BR.load(f"{REPO}/{BR.WEIGHTS}"); log("model loaded")
    src = os.environ.get("SRC", f"{R2}/{M}/video.mp4"); cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if LOCAL else OFFSET); rows = []; buf = []
    def flush():
        for k, g in zip([k for k, _ in buf], BR.detect_many(model, [f for _, f in buf])):
            rows.extend([[k, round(x, 1), round(y, 1), round(c, 4)] for x, y, c in g[:10]])
        buf.clear()
    for k in range(N):
        if k % STRIDE:
            if not cap.grab(): break
            continue
        ok, f = cap.read()
        if not ok: break
        if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
        buf.append((k, f))
        if len(buf) >= 4: flush()
        if k % 1000 == 0 and k: log(f"frame {k}/{N}")
    flush(); cap.release()
    json.dump(rows, gzip.open(f"{WORK}/ball_cands_rfdetr_reym.json.gz", "wt"))
    REP.update(ok=True, frames=N, guess_rows=len(rows)); save(); log("done")
except Exception:
    log(traceback.format_exc()); REP.update(ok=False, error=traceback.format_exc()[-3000:]); save()
