# Kaggle (free GPU), P1 30 Sep: the kit step's OWN input on every tracking piece, so the striped-kit fix can be tried offline.
# P1 v2 worked on the 24 kitprobe frames (Spånga 233/279) but never switched on in the real 60-s pieces (the pieces learn kits
# from 36 frames of one minute, with tiled RF-DETR). For each piece of the P2 run (7 grounds x 3 spots): the same 36 frames
# tracktest samples, RF-DETR people (same tiles), and per person a small crop (box + the strip below the feet) with the
# frame's grass colour and pitch-edge row -> /kaggle/working/<match>_<start>.pkl.gz. Also the current kit fit on the real
# frames (its log line + each person's reading) to check that the offline copy gives the same answer. No tracking.
# People under 22 px tall are left out (the kit step skips them). Dry run: DRY_RUN=1 DRY_CLIP=<mp4> (stand-in detector).
import subprocess, os, sys, json, time, gzip, pickle
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
t0 = time.time(); R2 = "{{R2}}"; W = "/kaggle/working"; DRY = os.environ.get("DRY_RUN") == "1"
if not DRY:
    sh("git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git /kaggle/temp/ia")
    sh("pip install -q rfdetr==1.11.0 supervision"); sys.path.insert(0, "/kaggle/temp/ia")
    sh("cd /kaggle/temp/ia && git log -1 --format='repo at %h %s'")
else: sys.path.insert(0, os.getcwd()); W = os.environ.get("DRY_OUT", "/tmp/kitcrops_dry"); os.makedirs(W, exist_ok=True)
import cv2, numpy as np
from ipanema import kits as K
sys.path.insert(0, os.path.join(sys.path[0], "tools")); from p1crops import pack
MATCHES = os.environ.get("P1_MATCHES", "p15u-vs-spanga-2026-09-25,p15u-vs-reymersholm-2026-09-18,p15u-vs-vasalund-2026-09-20,p15u-vs-djursholm-2026-09-26,"
                         "solberga-vs-p09-norrviken-2026-09-11,p09-norrviken-vs-solheim-2026-08-30,SFKBP1109_s1200").split(",")
DUR = float(os.environ.get("P1_DUR", "60")); FRACS = (0.2, 0.38, 0.72); M_FR = 36          # as players_all.py / tracktest.py
if DRY:
    class Det:                                                                   # stand-in detector: fixed boxes
        def detect(self, f): return [np.array([x, 400, x + 30, 480], float) for x in range(100, 1800, 150)]
    det = Det(); detect = det.detect
else:
    from ipanema import tracking as TR
    det = TR.RFDetrPerson("medium"); detect = lambda f: [b for b in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]
info = {}
for M in MATCHES:
    c = cv2.VideoCapture(f"{R2}/{M}/video.mp4" if not DRY else os.environ["DRY_CLIP"]); n_all = c.get(cv2.CAP_PROP_FRAME_COUNT); fps = c.get(cv2.CAP_PROP_FPS) or 29.97
    if n_all < 100: print("not reachable", M); info[M] = "not reachable"; continue
    starts = [20, 120, 220] if M == "SFKBP1109_s1200" else [int(n_all / fps * fr) for fr in FRACS]
    if DRY: starts = [0]
    for s in starts:
        tag = f"{M}_{s}"; t1 = time.time(); k0 = int(s * fps); n = int(DUR * fps); fb, frames = [], []
        for j in np.linspace(0, n - 1, M_FR).astype(int):
            c.set(cv2.CAP_PROP_POS_FRAMES, k0 + int(j)); ok, f = c.read()
            if not ok: continue
            bx = [b for b in detect(f) if b[3] - b[1] >= 22]; fb.append((f, np.array(bx))); frames.append(pack(f, bx))
            if len(frames) in (1, 18) and M.startswith(("p15u-vs-spanga", "SFKBP")): cv2.imwrite(f"{W}/{tag}_frame{len(frames):02d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        lines = []; tm = K.KitTeamModel().fit_frames(fb, log=lambda m: lines.append(m))
        tm.offpitch = M != "SFKBP1109_s1200"
        reading = [tm.predict_batch(f, b) for f, b in fb]
        with gzip.open(f"{W}/{tag}.pkl.gz", "wb") as fh:
            pickle.dump({"match": M, "start": s, "fps": fps, "frames": frames, "kaggle_fit_log": lines, "kaggle_reading": reading,
                         "kaggle_choice": tm.choice, "kaggle_missed": tm.choice_missed, "kaggle_pair_stat": [tm.pair, tm.stat]}, fh)
        info[tag] = {"frames": len(frames), "people": sum(len(x["people"]) for x in frames), "fit": lines, "minutes": round((time.time() - t1) / 60, 1)}
        print(tag, json.dumps(info[tag])[:600], flush=True)
        json.dump({"pieces": info, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{W}/run_info.json", "w"), indent=1)
    c.release()
print("done", round((time.time() - t0) / 60, 1), "min")
