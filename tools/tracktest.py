"""free runner: 20 s of live play from the SFK-BP clip, tracked with the old detector (YOLO football, AGPL) and RF-DETR
(Apache), SAME per-match kit step, SAME clean-up + gap filling. Counts, track lengths and pictures -> results/qa/tracktest/"""
import os, sys, json, time, subprocess
if not os.environ.get("LOCAL_CLIP"): subprocess.run("pip install -q ultralytics rfdetr torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True)
if not os.path.exists("player.pt") and not os.environ.get("LOCAL_CLIP"): subprocess.run(["gdown", "-q", "-O", "player.pt", "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"])
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from ipanema import tracking as TR, kits as K, linecal as LC
OUT = "results/qa/tracktest"; os.makedirs(OUT, exist_ok=True); START_S, DUR_S = float(os.environ.get("START_S", "60")), float(os.environ.get("DUR_S", "20"))
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True); open(f"{OUT}/log.txt", "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
src = os.environ.get("LOCAL_CLIP") or (os.environ.get("R2_PUBLIC_URL", "").rstrip("/") + "/SFKBP1109_s1200/video.mp4")
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); log(f"clip {src[-45:]}: {n_all} frames @ {fps:.2f}")
if n_all < 100: log("clip not reachable on R2"); sys.exit(1)
k0 = int(START_S * fps); n = int(DUR_S * fps); cap.set(cv2.CAP_PROP_POS_FRAMES, k0); piece = "/tmp/piece.mp4"
w = None
for i in range(n):
    ok, f = cap.read()
    if not ok: break
    if w is None: w = cv2.VideoWriter(piece, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
    w.write(f)
w.release(); cap.release()
rows = LC.find_rows(os.getcwd(), "SFKBP1109_s1200")
cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200 + START_S, log=log); H = cal["H"]; L, W = cal["L"], cal["W"]
def sample_frames(detect, m=12):
    c = cv2.VideoCapture(piece); out = []
    for j in np.linspace(0, n - 1, m).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(j)); ok, f = c.read()
        if ok: out.append((f, detect(f)))
    return out
rep = {}
for name in ("yolo", "rfdetr"):
    os.environ["IPANEMA_DETECTOR"] = name; t0 = time.time()
    if name == "rfdetr": det = TR.RFDetrPerson("medium"); detect = lambda f: [b for b in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]
    else:
        from ultralytics import YOLO; y = YOLO("player.pt")
        def detect(f):
            d, nm = TR.detect_tiled_batch(y, [f], 0.3, TR.FOLLOW_TILES, imgsz=960, half=False)[0]; ball = next((i for i, v in nm.items() if v.lower() == "ball"), -1)
            return [b for b, c in zip(d.xyxy, d.class_id) if int(c) != ball]
    tm = K.KitTeamModel().fit_frames(sample_frames(detect), log=log)
    per, _ = TR.track(piece, "player.pt", H, tm, 0.3, log=log, tiles=TR.FOLLOW_TILES, imgsz=960, pano=False)
    per = {k: v for k, v in per.items()}; raw_rows = sum(len(v) for v in per.values())
    per, cl = TR.clean(per, L, W, fps, log=log); per, nf = TR.fill_gaps(per, fps, 1.0, H=H)
    ids = {}
    for k, rs in per.items():
        for r in rs:
            if len(r) <= 6: ids.setdefault(r[0], []).append(k)
    tl = [len(v) / fps for v in ids.values()]
    med = {t: float(np.median([sum(1 for r in per[k] if r[1] == t and len(r) <= 6) for k in per])) for t in "AB"}
    medf = {t: float(np.median([sum(1 for r in per[k] if r[1] == t) for k in per])) for t in "AB"}
    rep[name] = {"minutes": round((time.time() - t0) / 60, 1), "observed_per_frame": med, "with_filled_per_frame": medf, "tracks": len(ids), "median_track_s": round(float(np.median(tl)), 2), "filled_rows": nf}
    log(f"{name}: {rep[name]}")
    c = cv2.VideoCapture(piece)
    for j in (0, n // 3, 2 * n // 3, n - 1):
        c.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = c.read()
        if not ok: continue
        for r in per.get(j, []):
            if r[3] is None: continue
            x, y_ = int(r[3][0]), int(r[3][1]); col = (0, 255, 255) if len(r) > 6 else ((0, 0, 255) if r[1] == "A" else (255, 128, 0))
            cv2.circle(f, (x, y_), 16, col, 3); cv2.putText(f, str(r[0]), (x + 12, y_ + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        t = f"{name} f{j}: A {sum(r[1]=='A' for r in per.get(j, []))} B {sum(r[1]=='B' for r in per.get(j, []))}"
        cv2.putText(f, t, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8); cv2.putText(f, t, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.imwrite(f"{OUT}/{name}_f{j:04d}.jpg", cv2.resize(f, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for t in ("A", "B"): cv2.imwrite(f"{OUT}/{name}_kit_{t}.png", tm.strips[t])
    json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)
