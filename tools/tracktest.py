"""free runner: 20 s of live play from the SFK-BP clip, tracked with the old detector (YOLO football, AGPL) and RF-DETR
(Apache), SAME per-match kit step, SAME clean-up + gap filling. Counts, track lengths and pictures -> results/qa/tracktest/"""
import os, sys, json, time, subprocess
if not os.environ.get("LOCAL_CLIP") and not os.environ.get("SKIP_INSTALL"): subprocess.run("pip install -q ultralytics rfdetr torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True)
if not os.path.exists("player.pt") and not os.environ.get("LOCAL_CLIP") and not os.environ.get("SKIP_INSTALL"): subprocess.run(["gdown", "-q", "-O", "player.pt", "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"])
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from ipanema import tracking as TR, kits as K, linecal as LC
OUT = "results/qa/tracktest"; os.makedirs(OUT, exist_ok=True); START_S, DUR_S = float(os.environ.get("START_S", "60")), float(os.environ.get("DUR_S", "20"))
_last = [0.0]
def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True); open(f"{OUT}/log.txt", "a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
    if os.environ.get("GITHUB_ACTIONS") and time.time() - _last[0] > 300:                  # 28 Sep: progress visible while it runs
        _last[0] = time.time(); subprocess.run(f"git add {OUT}/log.txt && git -c user.name=free-bot -c user.email=bot@ipanema commit -qm 'tracktest progress' && git pull -q --rebase origin main && git push -q origin main", shell=True)
MATCH = os.environ.get("MATCH", "SFKBP1109_s1200"); SFK = MATCH == "SFKBP1109_s1200"
if not SFK: OUT = f"results/qa/tracktest_{MATCH}"; os.makedirs(OUT, exist_ok=True)
src = os.environ.get("LOCAL_CLIP") or (os.environ.get("R2_PUBLIC_URL", "").rstrip("/") + f"/{MATCH}/video.mp4")
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
if SFK:
    rows = LC.find_rows(os.getcwd(), "SFKBP1109_s1200")
    cal = LC.calibration_for_clip(rows[2], n, fps, 1920, 1080, offset_s=1200 + START_S, log=log); H = cal["H"]; L, W = cal["L"], cal["W"]
else:                                                                           # ground not calibrated yet: pixels scaled to a pitch-sized box
    L, W = 106.0, 64.0; S_ = np.array([[1920 / L, 0, 0], [0, 1080 / W, 0], [0, 0, 1.0]]); H = {k: S_ for k in range(n)}
    log("no calibration for this ground: positions are screen positions (checks detection, kits, tracking only)")
def sample_frames(detect, m=12):
    c = cv2.VideoCapture(piece); out = []
    for j in np.linspace(0, n - 1, m).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(j)); ok, f = c.read()
        if ok: out.append((f, detect(f)))
    return out
os.environ.setdefault("IPANEMA_DET_BATCH", "1"); os.environ.setdefault("IPANEMA_LOG_EVERY", "50")
rep = {}; ROWS = {}
KEYS = [int(x) for x in np.linspace(0, n - 1, 8)]
def metrics(rows_by_k, dark_label):
    """rows_by_k: {k: [(id, team, px, filled)]} -> per-frame medians by kit (dark/light), tracks, median track length"""
    lab = lambda t: "dark" if t == dark_label else "light"
    obs = {t: float(np.median([sum(1 for r in rows_by_k.get(k, []) if lab(r[1]) == t and not r[3]) for k in range(n)])) for t in ("dark", "light")}
    allr = {t: float(np.median([sum(1 for r in rows_by_k.get(k, []) if lab(r[1]) == t) for k in range(n)])) for t in ("dark", "light")}
    ids = {}
    for k in range(n):
        for r in rows_by_k.get(k, []):
            if not r[3]: ids.setdefault((r[0], lab(r[1])), []).append(k)
    tl = {t: [len(v) / fps for (i, tt), v in ids.items() if tt == t] for t in ("dark", "light")}
    return {"observed_per_frame": obs, "shown_per_frame": allr, "tracks": len(ids), "median_track_s": {t: round(float(np.median(v)), 2) if v else None for t, v in tl.items()}}
if SFK:
    md = json.load(open("results/volume/runs/matches/SFKBP1109_s1200/match_data.json")); prior_dark = md["teams"]["dark"]
    ROWS["before (app, 27 Sep)"] = ({k - k0: [(p["id"], p["team"], p["px"], False) for p in md["frames"][k]["players"]] for k in range(k0, k0 + n)}, prior_dark)
    rep["before (app, 27 Sep)"] = metrics(*ROWS["before (app, 27 Sep)"]); log(f"before: {rep['before (app, 27 Sep)']}")
c = cv2.VideoCapture(piece)
for j in KEYS:
    c.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = c.read()
    if ok: cv2.imwrite(f"{OUT}/raw_{j:04d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])
BOXH = {}; RAW = {}
for name in [x for x in os.environ.get("DETECTORS", "rfdetr").split(",") if x]:
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
    label = {"yolo": "new, old detector", "rfdetr": "new, RF-DETR"}[name]
    ROWS[label] = ({k: [(r[0], r[1], None if r[3] is None else [float(r[3][0]), float(r[3][1])], len(r) > 6) for r in per.get(k, [])] for k in range(n)}, "A")   # new: A = darker kit
    BOXH[label] = {k: [None if r[4] is None else round(float(r[4][3] - r[4][1]), 1) for r in per.get(k, [])] for k in range(n)}   # box heights (px), same order
    RAW[label] = {k: [TR.RAW_TEAM.get((k, r[0])) for r in per.get(k, [])] for k in range(n)}   # P6: this frame's own colour reading, same order
    rep[label] = dict(metrics(*ROWS[label]), minutes=round((time.time() - t0) / 60, 1), filled_rows=nf); name = label
    log(f"{name}: {rep[name]}")
    for t in ("A", "B"): cv2.imwrite(f"{OUT}/{name.replace(',', '').replace(' ', '_')}_kit_{t}.png", tm.strips[t])
    json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)

json.dump({v: {str(k): r for k, r in rows.items() if k in KEYS} for v, (rows, _) in ROWS.items()}, open(f"{OUT}/rows_keyframes.json", "w"))
import gzip
if os.environ.get("SAVE_ALL_ROWS"):
    with gzip.open(f"{OUT}/rows_all.json.gz", "wt") as fh: json.dump({"k0": k0, "fps": fps, "rows": {v: {str(k): r for k, r in rows.items()} for v, (rows, _) in ROWS.items()},
                   "box_h": {v: {str(k): h for k, h in hh.items()} for v, hh in BOXH.items()},
                   "raw_team": {v: {str(k): h for k, h in hh.items()} for v, hh in RAW.items()}}, fh)
for j in KEYS:
    f0 = cv2.imread(f"{OUT}/raw_{j:04d}.jpg")
    if f0 is None: continue
    panels = []
    for v, (rows, dark) in ROWS.items():
        f = f0.copy(); nd = nl = 0
        for pid, team, px, filled in rows.get(j, []):
            if px is None: continue
            d = team == dark; nd += d; nl += not d; col = (0, 255, 255) if filled else ((0, 0, 255) if d else (255, 128, 0))
            cv2.circle(f, (int(px[0]), int(px[1])), 18, col, 4)
        t = f"{v}: dark {nd}  light {nl}"
        cv2.putText(f, t, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10); cv2.putText(f, t, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        panels.append(cv2.resize(f, (1280, 720)))
    cv2.imwrite(f"{OUT}/compare_{j:04d}.jpg", np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
lines = ["| | " + " | ".join(rep) + " |", "|---" * (len(rep) + 1) + "|"]
for key, f in (("dark players per frame (seen)", lambda r: r["observed_per_frame"]["dark"]), ("light players per frame (seen)", lambda r: r["observed_per_frame"]["light"]),
               ("dark per frame incl. filled", lambda r: r["shown_per_frame"]["dark"]), ("light per frame incl. filled", lambda r: r["shown_per_frame"]["light"]),
               ("dark: median time a player stays tracked (s)", lambda r: r["median_track_s"]["dark"]), ("light: median time tracked (s)", lambda r: r["median_track_s"]["light"]), ("track ids in 20 s", lambda r: r["tracks"])):
    lines.append(f"| {key} | " + " | ".join(str(f(r)) for r in rep.values()) + " |")
open(f"{OUT}/comparison.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)
