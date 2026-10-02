"""29 Sep: the RF-DETR ball finder (trained free on Kaggle on ~5,000 approved balls from 4 matches + SFK-BP clicks;
exam 84/108 vs 70/108 for the click model, ball found 74/81 vs 47/81). Needs the rfdetr package, so it runs in the
RF-DETR Modal image (players_rf / ball_rf), not the main image. Output: the pipeline's {frame: [(x, y, conf)]} shape.
Detection exactly as graded on Kaggle: the 1920x1080 frame cut into 4x2 overlapping 640 px tiles, guesses >= floor,
duplicates within 12 px dropped."""
import os, time, pickle, numpy as np

WEIGHTS = "results/kaggle/ballfinder_rfdetr/rfdetr_ball_small_20260929_fp16.pth"
VERSION = "rfdetr_ball_20260929"
FUSE_WASB = "1"          # 29 Sep free picker test, 34 app moments: RF-DETR + WASB 26, RF-DETR alone 25, old app setup 23
CROP, TX, TY = 640, [0, 427, 853, 1280], [0, 440]


def load(weights, device=None):
    """our fp16 file {"model": state_dict, "size", "resolution", "classes"} -> an rfdetr model ready for predict()"""
    import torch, rfdetr, tempfile
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ck["model"].items()}
    tmp = os.path.join(tempfile.gettempdir(), "rfdetr_ball_fp32.pth"); torch.save({"model": sd}, tmp)
    Model = {"small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase, "nano": rfdetr.RFDETRNano}[ck.get("size", "small")]
    kw = dict(pretrain_weights=tmp, num_classes=len(ck.get("classes", ["ball"])), resolution=int(ck.get("resolution", 640)))
    if device: kw["device"] = device
    m = Model(**kw)
    try: m.optimize_for_inference()
    except Exception: pass
    return m


def tiles(w, h):
    return [(x, y) for y in [min(y, h - CROP) for y in TY] for x in [min(x, w - CROP) for x in TX]]


def detect_many(model, frames, floor=0.05):
    """frames: list of BGR 1920x1080 images -> list of guess lists, one per frame, best first"""
    if not frames: return []
    h, w = frames[0].shape[:2]; tl = tiles(w, h); crops = []
    for f in frames:
        rgb = f[:, :, ::-1]; crops += [np.ascontiguousarray(rgb[y:y + CROP, x:x + CROP]) for x, y in tl]
    res = model.predict(crops, threshold=floor)
    if not isinstance(res, list): res = [res]
    out = []
    for i in range(len(frames)):
        det = []
        for (x, y), d in zip(tl, res[i * len(tl):(i + 1) * len(tl)]):
            for (a, b, c, e), cf in zip(d.xyxy, d.confidence): det.append((float(x + (a + c) / 2), float(y + (b + e) / 2), float(cf)))
        det.sort(key=lambda z: -z[2]); keep = []
        for z in det:
            if all(np.hypot(z[0] - k[0], z[1] - k[1]) > 12 for k in keep): keep.append(z)
        out.append(keep[:20])
    return out


def candidates_model(model, video, floor=0.05, batch=4, log=print, max_frames=0):
    """2 Oct: candidates for every frame with an already loaded model (no cache): {frame: [(x, y, conf)]}"""
    import cv2
    out = {}; t0 = time.time(); cap = cv2.VideoCapture(video); k = 0; buf = []
    def flush():
        for (kk, _), g in zip(buf, detect_many(model, [f for _, f in buf], floor)): out[kk] = g
        buf.clear()
    while True:
        if max_frames and k >= max_frames: break
        ok, f = cap.read()
        if not ok: break
        if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
        buf.append((k, f))
        if len(buf) >= batch: flush()
        if k % 1000 == 0 and k: log(f"  ball (RF-DETR): frame {k}, {k / max(1e-6, time.time() - t0):.1f} frames/s")
        k += 1
    flush(); return out

def candidates(video, weights, cache, floor=0.05, batch=4, log=print, max_frames=0):
    """every frame of the video -> {frame: [(x, y, conf)]}, cached, resumable (partial file every 1000 frames)"""
    import cv2
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    partial = cache + ".partial"; out = pickle.load(open(partial, "rb")) if os.path.exists(partial) else {}
    if out: log(f"  ball (RF-DETR): resuming, {len(out)} frames done")
    model = load(weights); t0 = time.time(); cap = cv2.VideoCapture(video); k = 0; buf = []
    def flush():
        for (kk, _), g in zip(buf, detect_many(model, [f for _, f in buf], floor)): out[kk] = g
        buf.clear()
    while True:
        if max_frames and k >= max_frames: break
        if k in out:
            if not cap.grab(): break
        else:
            ok, f = cap.read()
            if not ok: break
            if f.shape[:2] != (1080, 1920): f = cv2.resize(f, (1920, 1080))
            buf.append((k, f))
            if len(buf) >= batch: flush()
        if k % 1000 == 0 and k:
            log(f"  ball (RF-DETR): frame {k}, {k / max(1e-6, time.time() - t0):.1f} frames/s"); pickle.dump(out, open(partial, "wb"))
        k += 1
    cap.release(); flush()
    if not max_frames: pickle.dump(out, open(cache, "wb"))
    if os.path.exists(partial) and not max_frames: os.remove(partial)
    return out
