"""V3 (5 Oct, free runner, CPU, no Modal): why is the player on the ball missing on Vallentuna (hard sun + deep shade)?
For each by-eye carrier moment (owner A/B in reference/<vall>/ball_key_graded.json + the b-moments of
results/review/vall/poss_misses.json) save the full frame and RF-DETR people boxes WITH confidences under 4 settings:
  base   = pipeline (conf 0.3, FOLLOW_TILES)
  low    = conf 0.1 on the same frame (thresholds 0.15 / 0.2 tried offline)
  gamma  = shade lifted (gamma 0.6), conf 0.1
  clahe  = local contrast (CLAHE on L), conf 0.1
Graded offline by tools/v3lab.py (is the carrier boxed? kept by the kit model?).

    R2_PUBLIC_URL=... python tools/v3_frames.py          -> results/qa/v3/carriers/{mNN.jpg, boxes.json, moments.json}
    DRY=1 LOCAL_VIDEO=x.mp4 python tools/v3_frames.py    (stand-in detector, local video, no network)"""
import os, sys, json, time, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRY = os.environ.get("DRY") == "1"
if not DRY:
    try: import rfdetr  # noqa
    except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import tracking as TR

M = "p15u-vs-vallentuna-2026-10-03-6cce"
OUT = os.environ.get("V3_OUT", "results/qa/v3/carriers"); R2 = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
SETTINGS = (("base", None, 0.3), ("low", None, 0.1), ("gamma", "gamma", 0.1), ("clahe", "clahe", 0.1))

def moments():
    K = json.load(open(f"reference/{M}/ball_key_graded.json"))["items"]
    out = [{"id": i["id"], "t": i["t"], "ball": i["ball_px"], "owner": i["owner"]} for i in K if i.get("owner") in ("A", "B") and i.get("ball_px")]
    seen = {m["id"] for m in out}
    for m in json.load(open("results/review/vall/poss_misses.json"))["moments"]:
        if m["id"] not in seen and m["owner"] in ("A", "B"): out.append({"id": m["id"], "t": m["t"], "ball": m["true_ball"], "owner": m["owner"]})
    return sorted(out, key=lambda m: m["t"])

def lift(f, kind):
    if kind == "gamma":
        lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)]).astype(np.uint8); return cv2.LUT(f, lut)
    if kind == "clahe":
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB); lab[..., 0] = cv2.createCLAHE(2.0, (16, 16)).apply(lab[..., 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return f

class StandIn:
    def detect_batch(self, fs, conf, tiles):
        import supervision as sv
        h, w = fs[0].shape[:2]; xy = np.array([[w * a, h * 0.5, w * a + 30, h * 0.5 + 70] for a in np.linspace(0.1, 0.85, 10)], float)
        c = np.linspace(0.12, 0.9, len(xy)); k = c >= conf
        return [(sv.Detections(xyxy=xy[k], confidence=c[k], class_id=np.zeros(int(k.sum()), int)), {0: "person"})]

def main():
    det = StandIn() if DRY else TR.RFDetrPerson("medium"); t0 = time.time()
    src = os.environ.get("LOCAL_VIDEO") or f"{R2}/{M}/video.mp4"; cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    if not cap.isOpened(): sys.exit(f"no video: {src}")
    os.makedirs(OUT, exist_ok=True); ms = moments(); boxes = {}
    if DRY: ms = [dict(m, t=1.0 + 0.5 * j) for j, m in enumerate(ms[:3])]
    for j, m in enumerate(ms):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(m["t"] * fps))); ok, f = cap.read()
        if not ok: print(m["id"], "no frame"); continue
        f = cv2.resize(f, (1920, 1080)); name = f"m{j:02d}.jpg"; m["file"] = name; boxes[name] = {}
        for s, kind, conf in SETTINGS:
            d = det.detect_batch([lift(f, kind)], conf, TR.FOLLOW_TILES)[0][0]
            boxes[name][s] = [[round(float(v), 1) for v in b] + [round(float(c), 3)] for b, c in zip(d.xyxy, d.confidence)]
        cv2.imwrite(f"{OUT}/{name}", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{time.time() - t0:6.0f}s {m['id']} t={m['t']:.1f} " + " ".join(f"{s} {len(boxes[name][s])}" for s, _, _ in SETTINGS), flush=True)
    json.dump(boxes, open(f"{OUT}/boxes.json", "w")); json.dump({"match": M, "fps": fps, "moments": ms, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/moments.json", "w"), indent=1)

if __name__ == "__main__":
    main()
