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
    try: m.optimize_for_inference(**_fp16())
    except Exception as e: print(f"rfdetr optimize_for_inference skipped: {e!r}", flush=True)
    return m

def _fp16():
    """F1 (16 Oct, Kaggle T4): IPANEMA_RF_FP16=1 -> half precision, no compile (the clip ran at 2.7 frames/s in fp32)"""
    import torch
    return {"compile": False, "dtype": torch.float16} if os.environ.get("IPANEMA_RF_FP16") == "1" and torch.cuda.is_available() else {}

def autocast():
    """F1 (16 Oct, speed probe on a T4): rfdetr 1.11 has no optimize_for_inference; torch.autocast fp16 gives the ball finder
    7.6 instead of 3.4 frames/s, WASB 6.1 instead of 3.8, same answers. On with IPANEMA_RF_FP16=1, a no-op otherwise / on CPU."""
    import torch, contextlib
    return torch.autocast("cuda", dtype=torch.float16) if os.environ.get("IPANEMA_RF_FP16") == "1" and torch.cuda.is_available() else contextlib.nullcontext()


def tiles(w, h):
    return [(x, y) for y in [min(y, h - CROP) for y in TY] for x in [min(x, w - CROP) for x in TX]]


EDGE_PX = 2.0             # N0-21e: a guess box within this many px of a tile's inner side counts as cut by the tile


def edge_rule():
    """N0-21e (21 Oct): IPANEMA_BALL_EDGE=1 drops guesses whose box touches a tile's INNER side (not the picture's edge):
    the finder fires on things the tile cuts in half (a shin, a line, a shoe at the crop edge), and every such spot lies
    >= 200 px inside a neighbouring tile, which sees a real ball there whole. 0 (default) = old."""
    return os.environ.get("IPANEMA_BALL_EDGE", "0") == "1"


def cut_by_tile(box, x, y, w, h, crop=CROP, px=EDGE_PX):
    """box (x1, y1, x2, y2) in tile pixels of the tile at (x, y) in a w x h picture -> True when it touches an inner side"""
    a, b, c, e = box
    return bool((x > 0 and a <= px) or (x + crop < w and c >= crop - px) or (y > 0 and b <= px) or (y + crop < h and e >= crop - px))


def detect_many(model, frames, floor=0.05, edge=None, raw=None):
    """frames: list of BGR 1920x1080 images -> list of guess lists, one per frame, best first.
    edge: drop guesses cut by a tile's inner side (None = IPANEMA_BALL_EDGE). raw: a list that receives, per frame, every
    tile guess before merging as (x, y, conf, cut) - for checks."""
    if not frames: return []
    edge = edge_rule() if edge is None else edge
    h, w = frames[0].shape[:2]; tl = tiles(w, h); crops = []
    for f in frames:
        rgb = f[:, :, ::-1]; crops += [np.ascontiguousarray(rgb[y:y + CROP, x:x + CROP]) for x, y in tl]
    with autocast(): res = model.predict(crops, threshold=floor)
    if not isinstance(res, list): res = [res]
    out = []
    for i in range(len(frames)):
        det = []
        for (x, y), d in zip(tl, res[i * len(tl):(i + 1) * len(tl)]):
            for (a, b, c, e), cf in zip(d.xyxy, d.confidence):
                det.append((float(x + (a + c) / 2), float(y + (b + e) / 2), float(cf), cut_by_tile((a, b, c, e), x, y, w, h)))
        if raw is not None: raw.append(list(det))
        out.append(merge(det, edge))
    return out


def merge(det, edge=False, r=12, top=20):
    """one frame's tile guesses [(x, y, conf, cut)] -> best first, duplicates within r px dropped, at most top;
    edge: guesses cut by a tile's inner side left out first"""
    if edge: det = [z for z in det if not z[3]]
    keep = []
    for z in sorted(det, key=lambda z: -z[2]):
        if all(np.hypot(z[0] - k[0], z[1] - k[1]) > r for k in keep): keep.append(tuple(z[:3]))
    return keep[:top]


def _to_native(guesses, w, h):
    """N0-21e: guesses are found on the frame resized to 1920x1080; the calibration, players and WASB use the video's own
    pixels, so scale back (no change for a 1920x1080 video)"""
    if (w, h) == (1920, 1080) or not w or not h: return guesses
    fx, fy = w / 1920.0, h / 1080.0
    return [(x * fx, y * fy, c) for x, y, c in guesses]


def candidates_model(model, video, floor=0.05, batch=4, log=print, max_frames=0):
    """2 Oct: candidates for every frame with an already loaded model (no cache): {frame: [(x, y, conf)]}"""
    import cv2
    out = {}; t0 = time.time(); cap = cv2.VideoCapture(video); k = 0; buf = []; vw, vh = int(cap.get(3)), int(cap.get(4))
    def flush():
        for (kk, _), g in zip(buf, detect_many(model, [f for _, f in buf], floor)): out[kk] = _to_native(g, vw, vh)
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
    model = load(weights); t0 = time.time(); cap = cv2.VideoCapture(video); k = 0; buf = []; vw, vh = int(cap.get(3)), int(cap.get(4))
    def flush():
        for (kk, _), g in zip(buf, detect_many(model, [f for _, f in buf], floor)): out[kk] = _to_native(g, vw, vh)
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
