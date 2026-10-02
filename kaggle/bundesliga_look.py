# Kaggle (free, 2 Oct): first look at Daniel's Bundesliga clips (Drive, link-shared). Is it broadcast TV with cuts, or one
# fixed/follow camera? 12 frames per clip: our player detector (RF-DETR medium, conf 0.3, pipeline tiles) and our RF-DETR
# ball finder, boxes drawn. Plus a cut count (big frame-to-frame changes) over the first 3 minutes.
import os, sys, json, subprocess, cv2, numpy as np
W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True)
subprocess.run('pip install -q "rfdetr==1.11.0" supervision gdown', shell=True)
sys.path.insert(0, f"{T}/ia"); os.environ["IPANEMA_DETECTOR"] = "rfdetr"
from ipanema import tracking as TR, ballrf as BR
CLIPS = {"bundesliga1": "1aQs8FTVbZm7ewmDzvaq658d1CdV5k9dR", "bundesliga2": "1_WwMUPtx7yw5jx5pZhn-6KeU8eiPNsJC"}
det = TR.RFDetrPerson("medium"); ball = BR.load(f"{T}/ia/{BR.WEIGHTS}")
rep = {}
for name, fid in CLIPS.items():
    dst = f"{T}/{name}.mp4"
    r = subprocess.run(["gdown", "--fuzzy", f"https://drive.google.com/file/d/{fid}/view", "-O", dst], capture_output=True, text=True)
    if not os.path.exists(dst): rep[name] = {"error": (r.stdout + r.stderr)[-500:]}; print(rep[name], flush=True); continue
    cap = cv2.VideoCapture(dst); fps = cap.get(5); n = int(cap.get(7)); w, h = int(cap.get(3)), int(cap.get(4))
    info = {"fps": round(fps, 2), "frames": n, "seconds": round(n / fps, 1), "size": [w, h], "bytes": os.path.getsize(dst)}
    # cuts: mean abs difference between frames 0.5 s apart over the first 3 minutes
    cuts = 0; prev = None; step = max(1, int(fps * 0.5))
    for k in range(0, min(n, int(180 * fps)), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
        if prev is not None and np.abs(g - prev).mean() > 40: cuts += 1
        prev = g
    info["cuts_first_3_min"] = cuts
    rows = []; tiles = []
    for j, t in enumerate(np.linspace(5, n / fps - 5, 12)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: continue
        f = cv2.resize(f, (1920, 1080)); d = det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0]; b = BR.detect_many(ball, [f])[0]
        g = f.copy()
        for (a, bb, c, e) in d.xyxy: cv2.rectangle(g, (int(a), int(bb)), (int(c), int(e)), (0, 255, 255), 2)
        for (x, y, cf) in b[:3]: cv2.circle(g, (int(x), int(y)), 14, (0, 0, 255), 3); cv2.putText(g, f"{cf:.2f}", (int(x) + 16, int(y)), 0, 0.7, (0, 0, 255), 2)
        hs = [float(e - bb) for (a, bb, c, e) in d.xyxy]
        rows.append({"t": round(float(t), 1), "people": len(d.xyxy), "ball_top_conf": round(b[0][2], 2) if b else None, "person_h_px_median": round(float(np.median(hs)), 1) if hs else None})
        cv2.putText(g, f"{name} t={t:.0f}s people {len(d.xyxy)} ball {rows[-1]['ball_top_conf']}", (10, 32), 0, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(g, (960, 540)))
    for s in range(0, len(tiles), 4):
        cv2.imwrite(f"{W}/{name}_sheet{s // 4}.jpg", np.vstack(tiles[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    rep[name] = {**info, "frames_checked": rows}; print(name, info, flush=True)
    for r_ in rows: print(" ", r_, flush=True)
    os.remove(dst)
json.dump(rep, open(f"{W}/result.json", "w"), indent=1)
