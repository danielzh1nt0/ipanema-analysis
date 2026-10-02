"""F1c (2 Oct, free runner, CPU, no Modal): save full-match frames + RF-DETR people boxes so shadow-aware kit readings
can be tried OFFLINE (like results/qa/kitprobe, but over a whole match, sun and shadow).

Per match: the 36 frames the match kit model is fitted on (modal_app.rf_kits sampling: 18 per half) -> fit/, and test
frames offset from them (none used for fitting) -> test/. Each folder: fNN.jpg + boxes.json {name: [[x1,y1,x2,y2], ...]}
and times.json {name: seconds}.

    R2_PUBLIC_URL=... python tools/f1c_frames.py               -> results/qa/f1c/frames/<match>/
    DRY=1 LOCAL_VIDEO=x.mp4 python tools/f1c_frames.py         (stand-in detector, a short local video, no network)"""
import os, sys, json, time, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = os.environ.get("DRY") == "1"
if not DRY:
    try: import rfdetr  # noqa
    except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import tracking as TR

OUT = os.environ.get("F1C_OUT", "results/qa/f1c/frames")
R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
MATCHES = [m for m in os.environ.get("F1C_MATCHES", "p15u-vs-aik-2026-09-21-bd09:30,SFKBP1109:18").split(",") if m]
t0 = time.time()

class StandIn:
    def detect_batch(self, fs, conf, tiles):
        import supervision as sv
        h, w = fs[0].shape[:2]; xy = np.array([[w * a, h * 0.5, w * a + 30, h * 0.5 + 70] for a in np.linspace(0.1, 0.85, 10)], float)
        return [(sv.Detections(xyxy=xy, confidence=np.ones(len(xy)), class_id=np.zeros(len(xy), int)), {0: "person"})]

det = StandIn() if DRY else TR.RFDetrPerson("medium")

def open_video(match):
    for src in ([os.environ["LOCAL_VIDEO"]] if os.environ.get("LOCAL_VIDEO") else [f"{R2}/{match}/video.mp4", f"{R2}/{match}/full.mp4", f"{R2}/{match}/video_cropped.mp4"]):
        cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cap.isOpened() and n > 0: return cap, n, cap.get(cv2.CAP_PROP_FPS) or 29.97
        cap.release()
    return None, 0, 29.97

def times(spans, n_test):
    fit = np.concatenate([np.linspace(a + 0.05 * (b - a), b - 0.05 * (b - a), 18) for a, b in spans])       # modal_app.rf_kits
    per = [n_test // 2, n_test - n_test // 2]
    test = np.concatenate([np.linspace(a + 0.03 * (b - a), b - 0.03 * (b - a), k) + 0.37 for (a, b), k in zip(spans, per)])
    return fit, test

def dump(match, n_test):
    cap, n, fps = open_video(match)
    if cap is None: print(match, "no video"); return {"error": "no video"}
    T = n / fps; pf = f"periods/{match}.json"
    spans = json.load(open(pf))["periods_s"] if os.path.exists(pf) and not DRY else [[0.1 * T, 0.45 * T], [0.55 * T, 0.9 * T]]
    out = {"spans": spans, "frames": {}}
    for part, ts in zip(("fit", "test"), times(spans, n_test)):
        d = f"{OUT}/{match}/{part}"; os.makedirs(d, exist_ok=True); boxes, tt = {}, {}
        for i, t in enumerate(ts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
            if not ok: continue
            b = [[round(float(v), 1) for v in x] for x in det.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0].xyxy]
            name = f"f{i:02d}.jpg"; cv2.imwrite(f"{d}/{name}", f, [cv2.IMWRITE_JPEG_QUALITY, 85]); boxes[name] = b; tt[name] = round(float(t), 2)
            print(f"{time.time() - t0:6.0f}s {match} {part} {i} t={t:.0f}s people {len(b)}", flush=True)
        json.dump(boxes, open(f"{d}/boxes.json", "w")); json.dump(tt, open(f"{d}/times.json", "w"))
        out["frames"][part] = len(boxes)
    cap.release(); return out

if __name__ == "__main__":
    summ = {}
    for m in MATCHES:
        name, k = (m.split(":") + ["30"])[:2]
        try: summ[name] = dump(name, int(k))
        except Exception as e:
            import traceback; summ[name] = {"error": traceback.format_exc()[-1500:]}; print(summ[name]["error"])
    summ["minutes"] = round((time.time() - t0) / 60, 1); os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1); print(json.dumps(summ))
