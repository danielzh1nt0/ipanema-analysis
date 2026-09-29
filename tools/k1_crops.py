"""K1 runner (29 Sep): WASB peaks (30 local maxima >= 0.05) around each ball Daniel checked on the 4 good matches, then
ball / not-ball 3-frame crops (ipanema/k1crops.py). Runs on Kaggle's free GPU (kaggle/ballcrops_k1.py); videos from R2.
    R2_PUBLIC_URL=... python tools/k1_crops.py OUT_DIR
Dry run (stand-in video, random WASB weights, 2 checks per match, frames remapped into the fake video):
    DRY=1 LOCAL_VIDEO=fake.mp4 python tools/k1_crops.py OUT_DIR"""
import os, sys, json, gzip, time, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import k1crops as K, wasb as WB

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/ball/k1"; os.makedirs(OUT, exist_ok=True)
DRY = os.environ.get("DRY") == "1"; WINDOW = int(os.environ.get("K1_WINDOW", "12")); ROOT = os.environ.get("WROOT", "/kaggle/temp/wroot")
TMP = os.environ.get("K1_TMP", "/kaggle/temp"); R2 = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
t0 = time.time(); LOG = []
def log(m): LOG.append(f"{time.time() - t0:7.0f}s {m}"); print(LOG[-1], flush=True)

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"; log(f"device {dev}")
if DRY:                                                         # stand-in: the WASB network with random weights
    sys.path.insert(0, f"{WB.WASB_DIR}/src"); import yaml
    from models import build_model; from omegaconf import OmegaConf
    net = build_model(OmegaConf.create({"model": yaml.safe_load(open(f"{WB.WASB_DIR}/src/configs/model/wasb.yaml"))})).to(dev).eval()
    weights = "random (dry run)"
else:
    try: WB.ensure(ROOT, log=log)
    except Exception as e: log(f"WASB weights download FAILED (Google Drive): {e!r}"); json.dump({"error": "wasb weights", "detail": repr(e)[:500], "log": LOG}, open(f"{OUT}/k1_summary.json", "w"), indent=1); sys.exit(1)
    net = WB._model(ROOT, dev, finetuned=False); weights = "WASB soccer (pretrained, Google Drive)"
boxes = WB.tile_boxes()

def wasb_peaks(f3):
    base = [WB.to_base(f) for f in f3]; h, w = f3[1].shape[:2]; fx, fy = w / WB.BASE[0], h / WB.BASE[1]
    x = torch.from_numpy(np.stack([WB.crop_stack(base, b) for b in boxes])).to(dev)
    with torch.no_grad():
        pred = net(x); pred = list(pred.values())[0] if isinstance(pred, dict) else (pred[0] if isinstance(pred, (list, tuple)) else pred)
        hms = torch.sigmoid(pred).float().cpu().numpy()
    if DRY:                                                     # random weights give flat maps: stand-in peaks = bright blobs + one fixed decoy
        assert hms.shape[0] == len(boxes)
        n, _, st, cen = cv2.connectedComponentsWithStats((f3[1].min(2) > 240).astype(np.uint8))
        return [(float(cen[i][0]), float(cen[i][1]), 0.6) for i in range(1, n) if st[i][4] < 200] + [(1500.0, 300.0, 0.3)]
    return [(px * fx, py * fy, sc) for px, py, sc in WB.tiled_local_peaks([hms[t, 1] for t in range(len(boxes))], boxes, 0.05, 30)]

def read_window(cap, k0, W):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, k0 - W - 1)); fr = {}
    for k in range(max(0, k0 - W - 1), k0 + W + 2):
        ok, f = cap.read()
        if not ok: break
        fr[k - k0] = f
    return fr

X, META, PEAKS, SUM = [], [], {}, {"window": WINDOW, "weights": weights, "matches": {}}
for m in K.GOOD:
    checks = K.checked_balls("results/review", m)
    if DRY: video = os.environ["LOCAL_VIDEO"]
    else:
        video = f"{TMP}/{m}.mp4"
        if not os.path.exists(video):
            r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", video, f"{R2}/{m}/video.mp4"], capture_output=True, text=True)
            if r.returncode: log(f"{m}: download failed {r.stderr[-300:]}"); SUM["matches"][m] = {"error": "download"}; continue
    cap = cv2.VideoCapture(video); nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"{m}: {len(checks)} checked balls, video {nfr} frames at {fps:.2f} fps, {os.path.getsize(video) / 1e9:.1f} GB")
    if DRY: checks = [dict(c, frame=30 + 40 * i, xy=(300.0 + 5 * (30 + 40 * i), 500.0)) for i, c in enumerate(checks[:2])]   # the stand-in video's ball
    n0 = len(X); on_spot = 0; lens = []; PEAKS[m] = {}
    for c in checks:
        fr = read_window(cap, c["frame"], WINDOW)
        pk = {dk: wasb_peaks((fr[dk - 1], fr[dk], fr[dk + 1])) for dk in range(-WINDOW, WINDOW + 1) if all(j in fr for j in (dk - 1, dk, dk + 1))}
        x, meta, fol = K.crops_for_check(fr, pk, c, m, WINDOW)
        X += x; META += meta; on_spot += fol[0][2] >= 0; lens.append(len(fol))
        PEAKS[m][c["frame"]] = {"check": c, "followed": {str(k): [round(v, 2) for v in p] for k, p in fol.items()},
                                "peaks": {str(k): [[round(a, 1), round(b, 1), round(s, 3)] for a, b, s in v] for k, v in pk.items()}}
    cap.release()
    lab = [mm[6] for mm in META[n0:]]
    SUM["matches"][m] = {"checked_balls": len(checks), "wasb_peak_on_checked_spot": on_spot, "frames_followed_mean": round(float(np.mean(lens)), 1) if lens else 0,
                         "crops": len(lab), "ball": lab.count(1), "not_ball": lab.count(0)}
    log(f"{m}: {SUM['matches'][m]}")
    img = K.sheet(np.stack(X[n0:]), META[n0:]) if len(X) > n0 else None
    if img is not None: cv2.imwrite(f"{OUT}/sheet_{m}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not DRY and os.path.exists(video): os.remove(video)
np.savez_compressed(f"{OUT}/k1_crops.npz", X=np.stack(X) if X else np.zeros((0, 3, 32, 32, 3), np.uint8), meta=np.array(META, dtype=object))
json.dump(PEAKS, gzip.open(f"{OUT}/k1_peaks.json.gz", "wt"))
lab = [mm[6] for mm in META]; SUM.update(crops=len(lab), ball=lab.count(1), not_ball=lab.count(0), minutes=round((time.time() - t0) / 60, 1),
                                         MB=round(os.path.getsize(f"{OUT}/k1_crops.npz") / 1e6, 1), log=LOG[-40:])
json.dump(SUM, open(f"{OUT}/k1_summary.json", "w"), indent=1); log(f"done: {len(lab)} crops, {lab.count(1)} ball")
