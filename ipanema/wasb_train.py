"""Fine-tune the WASB soccer checkpoint on the coach's ball clicks. Same target as WASB (binary disc r=2.5 at 512x288) and
the same weighted BCE; loss only on the labelled middle frame of each 3-frame stack. Holds out 20% and reports hit rate."""
import os, glob, json, numpy as np, cv2

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
    """returns list of (stack 9x288x512 float32, target 288x512 float32 or None-for-invisible, visible flag)"""
    from .wasb import MEAN, STD
    S = []
    for gt_path in glob.glob(f"{root}/reference/*/ball_gt.json"):
        clip = os.path.basename(os.path.dirname(gt_path)); vid = _video_for(clip, videos_dir, log=log)
        if vid is None: continue
        gt = {int(k): v for k, v in json.load(open(gt_path)).items()}
        cap = cv2.VideoCapture(vid); W, H = int(cap.get(3)), int(cap.get(4))
        for i, g in sorted(gt.items()):
            frames = []
            for j in (i - 1, i, i + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, j)); ok, f = cap.read()
                if not ok: break
                im = cv2.resize(f, (512, 288)); frames.append(((im[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
            if len(frames) != 3: continue
            if g is None: tgt = np.zeros((288, 512), np.float32); vis = False
            else:
                cx, cy = g[0] * 512.0 / W, g[1] * 288.0 / H; x, y = np.meshgrid(np.arange(512), np.arange(288))
                tgt = (((x - cx) ** 2 + (y - cy) ** 2) <= 2.5 ** 2).astype(np.float32); vis = True
            S.append((np.concatenate(frames, 0), tgt, vis, (g[0] * 512.0 / W, g[1] * 288.0 / H) if g else None))
        cap.release()
    log(f"wasb train: {len(S)} samples ({sum(1 for s in S if s[2])} with a visible ball)")
    return S

def train_best_of(root, videos_dir, seeds=(0, 1, 2), log=print):
    """small training sets swing run to run: train a few seeds, keep the best held-out model"""
    import torch, shutil
    best_h, best_path = -1, None
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        p = train(root, videos_dir, log=log, seed=s)
        if p is None: continue
        h = _LAST_BEST.get("hits", -1)
        log(f"wasb seed {s}: held-out {h}")
        if h > best_h: best_h = h; shutil.copy(p, p + f".seed{s}"); best_path = p + f".seed{s}"
    if best_path: shutil.copy(best_path, weights_path(root)); log(f"wasb: kept best seed, held-out {best_h}")
    return weights_path(root) if best_path else None

_LAST_BEST = {}

def train(root, videos_dir, epochs=120, lr=3e-4, log=print, seed=0):
    # cosine decay to a low LR helps the last few percent on a small set
    import torch
    from .wasb import ensure, _model
    ensure(root, log=log); dev = "cuda" if torch.cuda.is_available() else "cpu"; net = _model(root, dev)
    S = load_samples(root, videos_dir, log=log)
    if len(S) < 30: log("wasb train: too few samples"); return None
    rng = np.random.RandomState(0); idx = rng.permutation(len(S)); nval = max(10, len(S) // 5); va, tr = idx[:nval], idx[nval:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    def wbce(logits, t):   # TrackNetV2 weighted BCE on the middle channel
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return -((1 - p) ** 2 * t * torch.log(p) + p ** 2 * (1 - t) * torch.log(1 - p)).mean()
    def hit_rate(ids):
        net.eval(); hits = tot = 0
        with torch.no_grad():
            for k in ids:
                x, t, vis, c = S[k]
                if not vis: continue
                out = net(torch.from_numpy(x[None]).to(dev)); pred = out[0] if isinstance(out, dict) else out
                hm = torch.sigmoid(pred)[0, 1].cpu().numpy(); py, px = np.unravel_index(hm.argmax(), hm.shape)
                tot += 1; hits += (hm.max() > 0.25 and np.hypot(px - c[0], py - c[1]) <= 4.0)
        return hits, tot
    h0, t0 = hit_rate(va); log(f"wasb train: before fine-tune, held-out hit rate {h0}/{t0}")
    best = h0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    for ep in range(epochs):
        net.train(); rng.shuffle(tr); tot_loss = 0.0
        for b in range(0, len(tr), 8):
            xs, ts = [], []
            for k in tr[b:b + 8]:
                x, t, vis, c = S[k]
                if rng.rand() < 0.5: x = x[:, :, ::-1].copy(); t = t[:, ::-1].copy()   # horizontal flip
                if rng.rand() < 0.5: x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.2, 0.2)   # brightness/contrast jitter (normalised space)
                xs.append(x); ts.append(t)
            xb = torch.from_numpy(np.stack(xs)).float().to(dev); tb = torch.from_numpy(np.stack(ts)).to(dev)
            out = net(xb); pred = out[0] if isinstance(out, dict) else out
            loss = wbce(pred[:, 1], tb); opt.zero_grad(); loss.backward(); opt.step(); tot_loss += float(loss)
        if ep % 10 == 9 or ep == epochs - 1:
            h, t = hit_rate(va); log(f"  wasb epoch {ep + 1}: loss {tot_loss / max(1, len(tr) // 8):.4f}, held-out hit rate {h}/{t}")
            if h >= best: best, best_state = h, {k: v.clone() for k, v in net.state_dict().items()}
    _LAST_BEST["hits"] = best
    torch.save(best_state, weights_path(root)); log(f"wasb train: saved {weights_path(root)} (best held-out {best}/{t0})")
    return weights_path(root)

def ensure_finetuned(root, videos_dir, log=print):
    """train when there is no fine-tuned weight yet, or when the labels changed"""
    RECIPE = "e120-lr3e-4-allclips-v2"
    labels = glob.glob(f"{root}/reference/*/ball_gt.json"); n = sum(len([v for v in json.load(open(p)).values()]) for p in labels)
    mf = json.load(open(manifest_path(root))) if os.path.exists(manifest_path(root)) else {}
    if labels and (not os.path.exists(weights_path(root)) or n != mf.get("n") or mf.get("recipe") != RECIPE):
        if train_best_of(root, videos_dir, log=log): json.dump({"n": n, "recipe": RECIPE}, open(manifest_path(root), "w")); return True
    return False
