"""Ball candidates from WASB (Tarashima et al., BMVC 2023; MIT): a 3-frame HRNet heatmap model built for tiny fast balls.
Replaces the single-frame YOLO ball detector. Output has the same shape as ball.candidates(): {frame: [(x, y, score), ...]}."""
import os, sys, pickle, numpy as np, cv2

WASB_DIR = os.environ.get("WASB_DIR", "/content/WASB-SBDT"); WEIGHT_ID = "1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def weights_path(root): return os.path.join(root, "models", "wasb_soccer_best.pth.tar")

def ensure(root, log=print):
    """clone the repo and fetch the pretrained soccer weight if missing"""
    import subprocess
    if not os.path.isdir(f"{WASB_DIR}/src"): subprocess.run(f"git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git {WASB_DIR}", shell=True, check=True)
    w = weights_path(root)
    if not os.path.exists(w):
        os.makedirs(os.path.dirname(w), exist_ok=True); import gdown; gdown.download(id=WEIGHT_ID, output=w, quiet=True); log(f"wasb: downloaded weights -> {w}")
    return w

def _model(root, device):
    import torch, yaml
    sys.path.insert(0, f"{WASB_DIR}/src")
    from models import build_model
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"model": yaml.safe_load(open(f"{WASB_DIR}/src/configs/model/wasb.yaml"))})
    m = build_model(cfg); ck = torch.load(weights_path(root), map_location=device)
    m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck); return m.to(device).eval()

def _peaks(hm, thr=0.25, max_n=6):
    """weighted centroids of connected blobs above thr, strongest first"""
    out = []
    if hm.max() <= thr: return out
    _, th = cv2.threshold(hm, thr, 1, cv2.THRESH_BINARY); n, lab = cv2.connectedComponents(th.astype(np.uint8))
    for m in range(1, n):
        ys, xs = np.where(lab == m); w = hm[ys, xs]; out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    return sorted(out, key=lambda z: -z[2])[:max_n]

def candidates(video, root, cache, log=print, batch=8, thr=0.25):
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"; ensure(root, log=log); net = _model(root, device)
    cap = cv2.VideoCapture(video); W, H = int(cap.get(3)), int(cap.get(4)); sx, sy = W / 512.0, H / 288.0
    out = {}; buf = []; idx = []; k = 0; chunks = []; chunk_idx = []; stats = []
    def flush():
        nonlocal chunks, chunk_idx
        if not chunks: return
        x = torch.from_numpy(np.stack(chunks)).to(device)
        with torch.no_grad():
            pred = net(x); pred = pred[0] if isinstance(pred, (list, tuple, dict)) else pred
            if isinstance(pred, dict): pred = list(pred.values())[0]
            hms = torch.sigmoid(pred).cpu().numpy()          # (B, 3, 288, 512)
        stats.extend(float(hms[b, j].max()) for b in range(hms.shape[0]) for j in range(3))
        for b, fr in enumerate(chunk_idx):
            for j in range(3):
                out[fr[j]] = [(px * sx, py * sy, s) for px, py, s in _peaks(hms[b, j], thr)]
        chunks, chunk_idx = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        im = cv2.resize(f, (512, 288)); im = (im[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        buf.append(im.transpose(2, 0, 1)); idx.append(k); k += 1
        if len(buf) == 3:
            chunks.append(np.concatenate(buf, 0)); chunk_idx.append(idx); buf, idx = [], []
            if len(chunks) == batch: flush()
        if k % 1500 == 0: log(f"  wasb frame {k}")
    if buf:   # pad the tail to a full triple
        while len(buf) < 3: buf.append(buf[-1]); idx.append(idx[-1])
        chunks.append(np.concatenate(buf, 0)); chunk_idx.append(idx)
    flush(); cap.release()
    n_det = sum(1 for v in out.values() if v); q = np.percentile(stats, [50, 90, 99]) if stats else [0, 0, 0]
    log(f"wasb: {n_det}/{k} frames with a ball peak, {sum(len(v) for v in out.values())/max(1,k):.1f} peaks/frame; heatmap max p50/p90/p99 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
    pickle.dump(out, open(cache, "wb")); return out
