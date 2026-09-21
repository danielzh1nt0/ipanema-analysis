"""Fine-tune the WASB soccer checkpoint on the coach's ball clicks, at the same tile scale the detector runs at.
Frames are stored at base resolution (960x540, uint8); every epoch cuts a fresh random tile around each ball (plus
ball-free negatives), so the model sees the ball at ~6 px exactly as in inference. Target: WASB's binary disc r=2.5
on the middle frame; loss: TrackNetV2 weighted BCE. 20% of labels are held out and scored with full tiled inference."""
import os, glob, json, numpy as np, cv2

EVAL_SETS = {"SFKBP1109_s1200", "bundesliga1", "bundesliga2"}   # the ball-check clips: never train on them

def weights_path(root): return os.path.join(root, "models", "wasb_finetuned.pth")
def manifest_path(root): return os.path.join(root, "models", "wasb_train_manifest.json")

def _video_for(clip, videos_dir, log=print):
    import re as _re
    m = _re.match(r"^(.*\d)([a-z])$", clip)          # extra label sets for the same match: SFKBP1109b -> SFKBP1109
    if m and not glob.glob(f"{videos_dir}/{clip}*"): log(f"wasb train: labels {clip} -> video of {m.group(1)}"); clip = m.group(1)
    v = next((p for p in glob.glob(f"{videos_dir}/{clip}.*")), None)
    if v is None and os.path.isdir(f"{videos_dir}/{clip}"):
        vs = [p for p in glob.glob(f"{videos_dir}/{clip}/*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]; v = max(vs, key=os.path.getsize) if vs else None
    if v is None and os.environ.get("R2_PUBLIC_URL"):     # labelled clip not on this machine: fetch from R2 if it was ever exported
        try:
            import requests
            url = f"{os.environ['R2_PUBLIC_URL']}/{clip}/video.mp4"; r = requests.get(url, stream=True, timeout=600)
            if r.status_code == 200:
                os.makedirs(videos_dir, exist_ok=True); v = f"{videos_dir}/{clip}.mp4"; open(v, "wb").write(b"".join(r.iter_content(1 << 20))); log(f"wasb train: fetched {clip} from R2")
            else: log(f"wasb train: {clip} not on R2 ({r.status_code}); its labels are skipped")
        except Exception as e: log(f"wasb train: fetch {clip} failed: {e!r}")
    return v

def load_samples(root, videos_dir, log=print):
    """-> list of dicts {frames: 3 x (540, 960, 3) uint8 base frames, ball: (x, y) in base px or None}"""
    from .wasb import to_base, BASE
    S = []
    for gt_path in sorted(glob.glob(f"{root}/reference/*/ball_gt.json")):
        clip = os.path.basename(os.path.dirname(gt_path))
        if clip in EVAL_SETS: log(f"wasb train: {clip} held out as a test set"); continue
        vid = _video_for(clip, videos_dir, log=log)
        if vid is None: log(f"wasb train: no video for {clip}; skipped"); continue
        gt = {int(k): v for k, v in json.load(open(gt_path)).items()}
        cap = cv2.VideoCapture(vid); W, H = int(cap.get(3)), int(cap.get(4)); n_ok = 0
        for i, g in sorted(gt.items()):
            frames = []
            for j in (i - 1, i, i + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, j)); ok, f = cap.read()
                if not ok or f is None: break
                frames.append(to_base(f))
            if len(frames) != 3: continue
            ball = None if g is None else (g[0] * BASE[0] / W, g[1] * BASE[1] / H)
            S.append({"frames": frames, "ball": ball, "clip": clip}); n_ok += 1
        cap.release(); log(f"wasb train: {clip}: {n_ok} of {len(gt)} labels loaded ({vid.split('/')[-1]})")
    log(f"wasb train: {len(S)} samples ({sum(1 for x in S if x['ball'])} with a visible ball)")
    return S

def _disc(cx, cy):
    from .wasb import NET
    x, y = np.meshgrid(np.arange(NET[0]), np.arange(NET[1]))
    return (((x - cx) ** 2 + (y - cy) ** 2) <= 2.5 ** 2).astype(np.float32)

def random_crop(sample, rng, p_negative=0.3, margin=12):
    """one training example at tile scale: (9, 288, 512) float32 input, (288, 512) target"""
    from .wasb import tile_boxes, crop_stack, NET, BASE
    _, _, tw, th = tile_boxes()[0]
    W, H = BASE
    b = sample["ball"]
    if b is not None and rng.rand() >= p_negative:        # window that contains the ball, anywhere inside it
        bx, by = b
        x0 = rng.randint(int(max(0, bx - tw + margin)), int(max(0, min(W - tw, bx - margin))) + 1)
        y0 = rng.randint(int(max(0, by - th + margin)), int(max(0, min(H - th, by - margin))) + 1)
        x0 = int(min(max(0, x0), W - tw)); y0 = int(min(max(0, y0), H - th))
        tgt = _disc((bx - x0) * NET[0] / tw, (by - y0) * NET[1] / th)
    else:                                                  # window without the ball (or ball not visible)
        for _ in range(20):
            x0 = rng.randint(0, W - tw + 1); y0 = rng.randint(0, H - th + 1)
            if b is None or not (x0 - margin <= b[0] < x0 + tw + margin and y0 - margin <= b[1] < y0 + th + margin): break
        else:
            return random_crop(sample, rng, p_negative=0.0, margin=margin)
        tgt = np.zeros((NET[1], NET[0]), np.float32)
    x = crop_stack(sample["frames"], (x0, y0, tw, th))
    if rng.rand() < 0.5: x = x[:, :, ::-1].copy(); tgt = tgt[:, ::-1].copy()
    if rng.rand() < 0.5: x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
    return x.astype(np.float32), tgt

def tiled_predict(net, dev, frames_base):
    """full tiled inference on one triple -> merged peaks (base coords) for the middle frame"""
    import torch
    from .wasb import tile_boxes, crop_stack, tiled_heatmaps_to_peaks
    boxes = tile_boxes()
    x = torch.from_numpy(np.stack([crop_stack(frames_base, bx) for bx in boxes])).to(dev)
    with torch.no_grad():
        pred = net(x)
        if isinstance(pred, dict): pred = list(pred.values())[0]
        elif isinstance(pred, (list, tuple)): pred = pred[0]
        hm = torch.sigmoid(pred).float().cpu().numpy()[:, 1]
    return tiled_heatmaps_to_peaks(list(hm), boxes)

def hit_rate(net, dev, S, ids, tol_base=7.5):
    """a hit = the strongest detection lies within tol_base px (base scale; = 15 px on 1080p) of the click"""
    net.eval(); hits = tot = 0
    for k in ids:
        s = S[k]
        if s["ball"] is None: continue
        tot += 1; peaks = tiled_predict(net, dev, s["frames"])
        if peaks and np.hypot(peaks[0][0] - s["ball"][0], peaks[0][1] - s["ball"][1]) <= tol_base: hits += 1
    return hits, tot

_LAST_BEST = {}
_SAMPLES = {}

def train(root, videos_dir, epochs=40, lr=3e-4, log=print, seed=0, eval_every=2, patience=5):
    import torch
    from .wasb import ensure, _model
    ensure(root, log=log); dev = "cuda" if torch.cuda.is_available() else "cpu"; net = _model(root, dev, finetuned=False)
    if "S" not in _SAMPLES: _SAMPLES["S"] = load_samples(root, videos_dir, log=log)
    S = _SAMPLES["S"]
    if len(S) < 30: log("wasb train: too few samples"); return None
    split = np.random.RandomState(0).permutation(len(S)); nval = max(10, len(S) // 5); va, tr = split[:nval], split[nval:]   # same split for every seed
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    def wbce(logits, t):
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return -((1 - p) ** 2 * t * torch.log(p) + p ** 2 * (1 - t) * torch.log(1 - p)).mean()
    h0, t0 = hit_rate(net, dev, S, va); log(f"wasb train: pretrained, held-out hit rate {h0}/{t0}")
    best, best_state, stale = h0, {k: v.clone() for k, v in net.state_dict().items()}, 0
    for ep in range(epochs):
        net.train(); order = rng.permutation(tr); tot_loss = 0.0; nb = 0
        for b in range(0, len(order), 8):
            xs, ts = zip(*[random_crop(S[k], rng) for k in order[b:b + 8]])
            xb = torch.from_numpy(np.stack(xs)).to(dev); tb = torch.from_numpy(np.stack(ts)).to(dev)
            pred = net(xb)
            if isinstance(pred, dict): pred = list(pred.values())[0]
            elif isinstance(pred, (list, tuple)): pred = pred[0]
            loss = wbce(pred[:, 1], tb); opt.zero_grad(); loss.backward(); opt.step(); tot_loss += float(loss); nb += 1
        if ep % eval_every == eval_every - 1 or ep == epochs - 1:
            h, t = hit_rate(net, dev, S, va); log(f"  wasb epoch {ep + 1}: loss {tot_loss / max(1, nb):.6f}, held-out hit rate {h}/{t}")
            if h > best: best, best_state, stale = h, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= patience: log(f"  wasb early stop at epoch {ep + 1} (best {best}/{t})"); break
    _LAST_BEST["hits"] = best
    torch.save(best_state, weights_path(root)); log(f"wasb train: saved (best held-out {best}/{t0})")
    return weights_path(root)

def train_best_of(root, videos_dir, seeds=(0, 1, 2), log=print):
    """small training sets swing run to run: train a few seeds from the pretrained weights, keep the best held-out model"""
    import shutil
    best_h, best_path = -1, None
    for s in seeds:
        p = train(root, videos_dir, log=log, seed=s)
        if p is None: continue
        h = _LAST_BEST.get("hits", -1); log(f"wasb seed {s}: held-out {h}")
        if h > best_h: best_h = h; shutil.copy(p, p + f".seed{s}"); best_path = p + f".seed{s}"
    if best_path: shutil.copy(best_path, weights_path(root)); log(f"wasb: kept best seed, held-out {best_h}")
    return weights_path(root) if best_path else None

def ensure_finetuned(root, videos_dir, log=print):
    """train when there is no fine-tuned weight yet, or when the labels or the recipe changed"""
    RECIPE = "tiles2x2-base960-e40-lr3e-4-earlystop-v1"
    labels = glob.glob(f"{root}/reference/*/ball_gt.json"); n = sum(len(json.load(open(p))) for p in labels)
    mf = json.load(open(manifest_path(root))) if os.path.exists(manifest_path(root)) else {}
    if labels and (not os.path.exists(weights_path(root)) or n != mf.get("n") or mf.get("recipe") != RECIPE):
        if train_best_of(root, videos_dir, log=log): json.dump({"n": n, "recipe": RECIPE}, open(manifest_path(root), "w")); return True
    return False
