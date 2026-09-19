"""Second opinion on ball candidates: a small CNN on 48x48 crops, trained from the coach's ball clicks (positives) and
non-ball candidates the detector proposed (negatives). Rescoring multiplies detector confidence by the classifier's ball probability."""
import os, glob, json, pickle, numpy as np, cv2

def weights_path(root): return os.path.join(root, "models", "ballcls.pt")

def _crop(f, x, y, s=24):
    h, w = f.shape[:2]; x0, y0 = int(max(0, min(w - 2 * s, x - s))), int(max(0, min(h - 2 * s, y - s)))
    c = f[y0:y0 + 2 * s, x0:x0 + 2 * s]
    return cv2.resize(c, (48, 48)) if c.size else None

def _net():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 6 * 6, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

def build(root, videos_dir, log=print):
    """positives from every ball_gt.json; negatives from cached candidates > 40 px from the labelled ball in those frames (+ random candidates elsewhere)"""
    X, Y = [], []
    for gt_path in glob.glob(f"{root}/reference/*/ball_gt.json"):
        clip = os.path.basename(os.path.dirname(gt_path))
        vid = next((p for p in glob.glob(f"{videos_dir}/{clip}.*")), None)
        if vid is None and os.path.isdir(f"{videos_dir}/{clip}"):
            vs = [p for p in glob.glob(f"{videos_dir}/{clip}/*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]; vid = max(vs, key=os.path.getsize) if vs else None
        if vid is None: continue
        gt = {int(k): v for k, v in json.load(open(gt_path)).items() if v}
        cands = {}
        for cp in glob.glob(f"{root}/cache/*{clip}*/ball_cands.pkl"):
            try: cands.update(pickle.load(open(cp, "rb")))
            except Exception: pass
        cap = cv2.VideoCapture(vid)
        for i, (gx, gy) in gt.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            if not ok: continue
            for dx, dy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3), (2, 2), (-2, -2)):
                c = _crop(f, gx + dx, gy + dy)
                if c is not None: X.append(c); Y.append(1)
            negs = [(x, y) for x, y, _ in cands.get(i, []) if np.hypot(x - gx, y - gy) > 40][:12]
            if len(negs) < 6:   # add hard negatives near players / lines: random points on the frame
                negs += [(np.random.uniform(0, f.shape[1]), np.random.uniform(f.shape[0] * 0.3, f.shape[0])) for _ in range(6 - len(negs))]
            for x, y in negs:
                c = _crop(f, x, y)
                if c is not None: X.append(c); Y.append(0)
        cap.release()
    log(f"ball classifier dataset: {sum(Y)} positives, {len(Y) - sum(Y)} negatives")
    return np.array(X), np.array(Y)

def train(root, videos_dir, epochs=25, log=print):
    import torch, torch.nn as nn
    X, Y = build(root, videos_dir, log=log)
    if len(Y) < 100 or sum(Y) < 30: log("ball classifier: not enough data"); return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"; net = _net().to(dev)
    xt = torch.tensor(X[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0; yt = torch.tensor(Y).float()
    idx = np.random.permutation(len(Y)); nval = max(20, len(Y) // 5); va, tr = idx[:nval], idx[nval:]
    opt = torch.optim.Adam(net.parameters(), 1e-3); pos_w = torch.tensor([(len(Y) - sum(Y)) / max(1, sum(Y))]).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(tr)
        for b in range(0, len(perm), 64):
            j = perm[b:b + 64]; xb = xt[j].to(dev); yb = yt[j].to(dev)
            if np.random.rand() < 0.5: xb = torch.flip(xb, dims=[3])
            opt.zero_grad(); loss = lossf(net(xb).squeeze(1), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): p = torch.sigmoid(net(xt[va].to(dev)).squeeze(1)).cpu().numpy()
        acc = float(((p > 0.5) == (Y[va] > 0.5)).mean()); rec = float((p[Y[va] == 1] > 0.5).mean()) if (Y[va] == 1).any() else 0
        if ep % 5 == 4 or ep == epochs - 1: log(f"  ballcls epoch {ep + 1}: val acc {acc:.2f}, ball recall {rec:.2f}")
    torch.save(net.state_dict(), weights_path(root)); log(f"ball classifier -> {weights_path(root)}"); return weights_path(root)

def rescore(video, cands, root, cache=None, log=print):
    import torch
    if cache and os.path.exists(cache): return pickle.load(open(cache, "rb"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"; net = _net().to(dev); net.load_state_dict(torch.load(weights_path(root), map_location=dev)); net.eval()
    out = {}; cap = cv2.VideoCapture(video); k = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        cs = cands.get(k, [])
        if cs:
            crops = [_crop(f, x, y) for x, y, _ in cs]; keep = [i for i, c in enumerate(crops) if c is not None]
            if keep:
                xb = torch.tensor(np.array([crops[i][..., ::-1].copy() for i in keep])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.no_grad(): p = torch.sigmoid(net(xb).squeeze(1)).cpu().numpy()
                new = list(cs)
                for i, pi in zip(keep, p): x, y, cf = cs[i]; new[i] = (x, y, float(min(0.99, cf * (0.2 + 0.8 * pi) * 2.0)))
                out[k] = new
            else: out[k] = cs
        else: out[k] = cs
        k += 1
        if k % 2000 == 0: log(f"  ball classifier frame {k}")
    cap.release()
    if cache: pickle.dump(out, open(cache, "wb"))
    log("ball classifier: candidates rescored"); return out
