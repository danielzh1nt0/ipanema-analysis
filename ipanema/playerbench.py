"""28 Sep: player detection checked on ALL our footage (Daniel: 'accurate across ALL of our own footage').
For every match: n frames spread over the game; the tracking detector (same tiles, same size) is run ONCE at a low
threshold, and every box is drawn by the lowest setting that would keep it:
  green  = found today (confidence >= 0.30)
  yellow = found only if the bar is lowered to 0.15
  grey   = only at 0.10 (usually junk - shows what a lower bar would cost)
  magenta= called referee by teams.referee_kit
Claude grades each picture by eye (missed players, false boxes) - no clicking for Daniel."""
import base64, numpy as np, cv2

LEVELS = ((0.30, (0, 200, 0), "today"), (0.15, (0, 230, 255), "at 0.15"), (0.10, (160, 160, 160), "at 0.10"))

def pick_frames(n_frames, n=10, lo=0.08, hi=0.92):
    return [int(n_frames * (lo + (hi - lo) * (i + 0.5) / n)) for i in range(n)]

def classify(det, names, frame, referee_fn):
    """-> list of (x1, y1, x2, y2, conf, kind) with kind in today / at 0.15 / at 0.10 / referee / model-referee"""
    out = []; ref_id = next((i for i, nm in (names or {}).items() if "referee" in str(nm).lower()), None)
    for j in range(len(det)):
        b = det.xyxy[j]; c = float(det.confidence[j]); cls = int(det.class_id[j]) if det.class_id is not None else -1
        if cls == ref_id: kind = "model-referee"
        elif referee_fn(frame, b): kind = "referee"
        else: kind = next(lab for thr, _, lab in LEVELS if c >= thr) if c >= LEVELS[-1][0] else None
        if kind: out.append((*[float(v) for v in b], round(c, 3), kind))
    return out

def draw(frame, boxes, title, width=1280):
    s = width / frame.shape[1]; im = cv2.resize(frame, (width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    col = {lab: c for _, c, lab in LEVELS}; col["referee"] = (255, 0, 255); col["model-referee"] = (255, 0, 255)
    for x1, y1, x2, y2, c, kind in boxes:
        cv2.rectangle(im, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), col[kind], 2)
    n = {k: sum(b[5] == k for b in boxes) for k in col}
    t = f"{title} | today {n['today']} +0.15: {n['at 0.15']} +0.10: {n['at 0.10']} referee {n['referee'] + n['model-referee']}"
    cv2.putText(im, t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5); cv2.putText(im, t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im

def jpg(im, q=85): return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()
