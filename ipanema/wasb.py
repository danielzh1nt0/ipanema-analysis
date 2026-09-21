"""Ball candidates from WASB (Tarashima et al., BMVC 2023; MIT): a 3-frame HRNet heatmap model built for tiny fast balls.
Replaces the single-frame YOLO ball detector. Output has the same shape as ball.candidates(): {frame: [(x, y, score), ...]}."""
import os, sys, pickle, numpy as np, cv2

WASB_DIR = os.environ.get("WASB_DIR", "/content/WASB-SBDT"); WEIGHT_ID = "1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def weights_path(root): return os.path.join(root, "models", "wasb_soccer_best.pth.tar")

# Tiling: frames are halved (1920x1080 -> 960x540) and cut into 2x2 overlapping tiles of ~519x292, each fed at 512x288.
# That shows the ball at ~6 px, the size WASB was built for (a whole frame at 512x288 shrinks it to ~3 px).
BASE = (960, 540); TILES = (2, 2); OVERLAP = 0.15; NET = (512, 288); TAG = "t2x2"

def tile_boxes(W=BASE[0], H=BASE[1], tiles=TILES, overlap=OVERLAP):
    cols, rows = tiles
    tw = int(round(W / (cols - (cols - 1) * overlap))) if cols > 1 else W
    th = int(round(H / (rows - (rows - 1) * overlap))) if rows > 1 else H
    xs = [int(round(i * (W - tw) / (cols - 1))) for i in range(cols)] if cols > 1 else [0]
    ys = [int(round(j * (H - th) / (rows - 1))) for j in range(rows)] if rows > 1 else [0]
    return [(x, y, tw, th) for y in ys for x in xs]

def to_base(frame):
    return cv2.resize(frame, BASE, interpolation=cv2.INTER_AREA)

def crop_stack(frames_base, box):
    """three base frames -> one (9, 288, 512) float32 network input for the given tile box"""
    x, y, tw, th = box
    ims = [cv2.resize(f[y:y + th, x:x + tw], NET, interpolation=cv2.INTER_AREA) for f in frames_base]
    return np.concatenate([((im[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for im in ims], 0)

def merge_peaks(peaks, radius=6.0, max_n=6):
    """peaks in base coords from overlapping tiles -> strongest first, duplicates within radius removed"""
    out = []
    for x, y, sc in sorted(peaks, key=lambda z: -z[2]):
        if all((x - a) ** 2 + (y - b) ** 2 > radius ** 2 for a, b, _ in out): out.append((x, y, sc))
    return out[:max_n]

def tiled_heatmaps_to_peaks(hms_for_tiles, boxes, thr=0.25):
    """hms_for_tiles: list of (288, 512) heatmaps, one per box -> merged peaks in base coords"""
    peaks = []
    for hm, (x, y, tw, th) in zip(hms_for_tiles, boxes):
        for px, py, sc in _peaks(hm, thr):
            peaks.append((x + px * tw / NET[0], y + py * th / NET[1], sc))
    return merge_peaks(peaks)

def ensure(root, log=print):
    """clone the repo and fetch the pretrained soccer weight if missing"""
    import subprocess
    if not os.path.isdir(f"{WASB_DIR}/src"): subprocess.run(f"git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git {WASB_DIR}", shell=True, check=True)
    w = weights_path(root)
    if not os.path.exists(w):
        os.makedirs(os.path.dirname(w), exist_ok=True); import gdown; gdown.download(id=WEIGHT_ID, output=w, quiet=True); log(f"wasb: downloaded weights -> {w}")
    return w

def _model(root, device, finetuned=True):
    import torch, yaml
    sys.path.insert(0, f"{WASB_DIR}/src")
    from models import build_model
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"model": yaml.safe_load(open(f"{WASB_DIR}/src/configs/model/wasb.yaml"))})
    m = build_model(cfg)
    ft = os.path.join(root, "models", "wasb_finetuned.pth")
    if finetuned and os.path.exists(ft): m.load_state_dict(torch.load(ft, map_location=device))
    else:
        ck = torch.load(weights_path(root), map_location=device); m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    return m.to(device).eval()

def _peaks(hm, thr=0.25, max_n=6):
    """weighted centroids of connected blobs above thr, strongest first"""
    out = []
    if hm.max() <= thr: return out
    _, th = cv2.threshold(hm, thr, 1, cv2.THRESH_BINARY); n, lab = cv2.connectedComponents(th.astype(np.uint8))
    for m in range(1, n):
        ys, xs = np.where(lab == m); w = hm[ys, xs]; out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    return sorted(out, key=lambda z: -z[2])[:max_n]

def candidates(video, root, cache, log=print, batch=8, thr=0.25, videos_dir=None):
    import torch
    ensure(root, log=log)
    try:
        from .wasb_train import ensure_finetuned
        if ensure_finetuned(root, videos_dir or os.path.join(root, "videos"), log=log):
            for c in [cache] + [p for p in os.listdir(os.path.dirname(cache)) if False]: pass
    except Exception as e: log(f"wasb fine-tune skipped: {e!r}")
    ft = os.path.join(root, "models", "wasb_finetuned.pth"); tag = str(int(os.path.getmtime(ft))) if os.path.exists(ft) else "pre"
    cache = cache.replace(".pkl", f"_{tag}_{TAG}.pkl")
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    device = "cuda" if torch.cuda.is_available() else "cpu"; net = _model(root, device); log(f"wasb: using {'fine-tuned' if tag != 'pre' else 'pretrained'} weights, {TILES[0]}x{TILES[1]} tiles")
    cap = cv2.VideoCapture(video); W, H = int(cap.get(3)), int(cap.get(4)); fx, fy = W / BASE[0], H / BASE[1]
    boxes = tile_boxes(); nt = len(boxes)
    out = {}; buf = []; idx = []; k = 0; stats = []; pend_x = []; pend_meta = []
    def flush():
        nonlocal pend_x, pend_meta
        if not pend_x: return
        x = torch.from_numpy(np.stack(pend_x)).to(device)
        with torch.no_grad():
            pred = net(x)
            if isinstance(pred, dict): pred = list(pred.values())[0]
            elif isinstance(pred, (list, tuple)): pred = pred[0]
            hms = torch.sigmoid(pred).float().cpu().numpy()          # (B*tiles, 3, 288, 512)
        for g, fr in enumerate(pend_meta):                             # one group of tiles per frame triple
            block = hms[g * nt:(g + 1) * nt]
            for j in range(3):
                stats.append(float(block[:, j].max()))
                peaks = tiled_heatmaps_to_peaks([block[t, j] for t in range(nt)], boxes, thr)
                out[fr[j]] = [(px * fx, py * fy, sc) for px, py, sc in peaks]
        pend_x, pend_meta = [], []
    def push_triple(frames_base, ids):
        for box in boxes: pend_x.append(crop_stack(frames_base, box))
        pend_meta.append(ids)
        if len(pend_meta) * nt >= batch * 4: flush()
    while True:
        ok, f = cap.read()
        if not ok: break
        buf.append(to_base(f)); idx.append(k); k += 1
        if len(buf) == 3: push_triple(buf, idx); buf, idx = [], []
        if k % 1500 == 0: log(f"  wasb frame {k}")
    if buf:
        while len(buf) < 3: buf.append(buf[-1]); idx.append(idx[-1])
        push_triple(buf, idx)
    flush(); cap.release()
    n_det = sum(1 for v in out.values() if v); q = np.percentile(stats, [50, 90, 99]) if stats else [0, 0, 0]
    log(f"wasb: {n_det}/{k} frames with a ball peak, {sum(len(v) for v in out.values())/max(1,k):.1f} peaks/frame; heatmap max p50/p90/p99 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    pickle.dump(out, open(cache, "wb")); return out
