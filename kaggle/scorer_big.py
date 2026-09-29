# Kaggle (free GPU, 29 Sep): "ball or not" judge trained on thousands of approved ball labels, graded like tools/scorerlab.py.
# Inputs from R2 (copied off Modal 29 Sep): labels/<match>_trainset_clicks.json, models/wasb_finetuned.pth, the match
# videos <match>/video.mp4, the full SFK-BP video SFKBP1109/video.mp4 and the clip SFKBP1109_s1200/video.mp4.
# Output: /kaggle/working/result.json (+ log). LOCAL=1 runs a dry run with stand-in files.
import os, sys, json, time, subprocess, urllib.request, numpy as np
R2 = os.environ.get("R2_BASE", "{{R2}}").rstrip("/"); LOCAL = os.environ.get("LOCAL") == "1"
WORK = os.environ.get("WORK", "/kaggle/working"); TMP = os.environ.get("TMPD", "/kaggle/temp"); os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True)
t0 = time.time(); LOG = []
def log(m): LOG.append(f"{time.time() - t0:7.0f}s {m}"); print(LOG[-1], flush=True); open(f"{WORK}/log.txt", "w").write("\n".join(LOG))
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True); log((r.stdout + r.stderr)[-500:].strip()); return r.returncode
def get(rel, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return dst
    req = urllib.request.Request(f"{R2}/{rel}", headers={"User-Agent": "Mozilla/5.0 (ipanema kaggle)"})   # R2 refused python's default agent (403)
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b: break
            f.write(b)
    return dst
MATCHES = ["p15u-vs-vasalund-2026-09-20", "p09-norrviken-vs-solheim-2026-08-30", "p15u-vs-spanga-2026-09-25", "p15u-vs-djursholm-2026-09-26"]
PER_MATCH = int(os.environ.get("PER_MATCH", "500")); MIN_CONF = float(os.environ.get("MIN_CONF", "0.3"))
try:
    REPO = os.environ.get("REPO_DIR", f"{TMP}/ia")
    if not LOCAL:
        sh("nvidia-smi --query-gpu=name --format=csv,noheader")
        sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}")
        sh(f"git clone -q --depth 1 https://github.com/nttcom/WASB-SBDT.git {TMP}/WASB-SBDT"); os.environ["WASB_DIR"] = f"{TMP}/WASB-SBDT"
        sh("pip install -q omegaconf hydra-core")
    sys.path.insert(0, REPO)
    import cv2, torch
    from ipanema import ballprobe as BP, ballscorer as SC
    root = f"{TMP}/root"; os.makedirs(f"{root}/models", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if LOCAL:
        wasb_fn = lambda f3: [(float(f3[1].shape[1] // 2), float(f3[1].shape[0] // 2), 0.9), (100.0, 100.0, 0.4), (300.0, 200.0, 0.2)]
    else:
        from ipanema import wasb as WB
        get("models/wasb_finetuned.pth", f"{root}/models/wasb_finetuned.pth")
        net = WB._model(root, dev); boxes = WB.tile_boxes(); log(f"WASB fine-tuned weights loaded on {dev}")
        def wasb_fn(f3):
            base = [WB.to_base(f) for f in f3]; h, w = f3[1].shape[:2]; fx, fy = w / WB.BASE[0], h / WB.BASE[1]
            x = torch.from_numpy(np.stack([WB.crop_stack(base, b) for b in boxes])).to(dev)
            with torch.no_grad():
                pred = net(x); pred = list(pred.values())[0] if isinstance(pred, dict) else (pred[0] if isinstance(pred, (list, tuple)) else pred)
                hms = torch.sigmoid(pred).float().cpu().numpy()
            return [(px * fx, py * fy, sc) for px, py, sc in WB.tiled_local_peaks([hms[t, 1] for t in range(len(boxes))], boxes, 0.05, 30)]
    def read3(cap, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, k - 1)); fr = []
        for _ in range(3):
            ok, f = cap.read()
            if ok: fr.append(f)
        if len(fr) < 2: return None
        while len(fr) < 3: fr.append(fr[-1])
        return tuple(fr)
    X, META = [], []
    def cut(src, moments, tag):
        cap = cv2.VideoCapture(src); n0 = len(X); t1 = time.time()
        for i, m in enumerate(sorted(moments, key=lambda z: z["frame"])):
            f3 = read3(cap, m["frame"])
            if f3 is None: continue
            for x, y, s, lab in BP.label_guesses(BP.merge(wasb_fn(f3), 6.0), m["truth"], n_neg=12):
                X.append(BP.crop3(f3, x, y, 32)); META.append([f"{tag}:{m['id']}", m["split"], m["frame"], round(x, 1), round(y, 1), round(float(s), 4), lab])
            if (i + 1) % 100 == 0: log(f"{tag}: {i + 1}/{len(moments)} frames, {len(X) - n0} crops, {time.time() - t1:.0f}s")
        cap.release(); log(f"{tag}: {len(X) - n0} crops ({sum(1 for m in META[n0:] if m[6] == 1)} ball)")
    rng = np.random.default_rng(0)
    # 1) approved labels from the 4 training matches (spread over the match; only moving-ball runs; no spare balls)
    for mt in MATCHES:
        try:
            if LOCAL: labs = json.load(open(f"{TMP}/fake_labels.json")); src = f"{TMP}/fake.mp4"
            else: labs = json.load(open(get(f"labels/{mt}_trainset_clicks.json", f"{TMP}/{mt}_labels.json")))["labels"]; src = f"{R2}/{mt}/video.mp4"
            good = [l for l in labs if l["conf"] >= MIN_CONF and l.get("run", 0) >= 4]
            pick = [good[i] for i in sorted(rng.choice(len(good), min(PER_MATCH, len(good)), replace=False))] if good else []
            log(f"{mt}: {len(labs)} labels, {len(good)} usable, using {len(pick)}")
            cut(src, [{"id": f"{mt}:{l['frame']}", "frame": int(l["frame"]), "truth": (l["x"], l["y"]), "split": "train"} for l in pick], mt[:12])
        except Exception as e: log(f"{mt}: skipped ({e!r})")
    # 2) SFK-BP clicks: train split + exam (full match), and the 34 clip moments
    clicks = json.load(open(f"{REPO}/results/labels/SFKBP1109_ball_clicks.json"))["frames"]
    full = f"{TMP}/fake.mp4" if LOCAL else f"{R2}/SFKBP1109/video.mp4"
    cap = cv2.VideoCapture(full); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if nfr < 100 and not LOCAL:
        raise RuntimeError("full SFK-BP video not readable from R2"); cap = cv2.VideoCapture(full); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    log(f"SFK-BP full video: {nfr} frames")
    cl = clicks[:6] if LOCAL else clicks
    cut(full, [{"id": r["file"], "frame": int(round(r["t"] * fps)), "truth": (r["x"], r["y"]) if r["x"] is not None else None, "split": r["split"]} for r in cl], "sfk")
    mom = json.load(open(f"{REPO}/results/picker/moments34.json"))
    clip = f"{TMP}/fake.mp4" if LOCAL else f"{R2}/SFKBP1109_s1200/video.mp4"
    cut(clip, [{"id": str(k), "frame": int(k), "truth": v["truth"], "split": "clip34"} for k, v in (list(mom.items())[:3] if LOCAL else mom.items())], "clip")
    X = np.stack(X); y = np.array([m[6] for m in META]); split = np.array([m[1] for m in META])
    tr = split == "train"; log(f"crops {len(X)}: train {int(tr.sum())} ({int(y[tr].sum())} ball), exam {int((split == 'exam').sum())}, clip {int((split == 'clip34').sum())}")
    sfk_only = np.array([m[0].startswith("sfk:") for m in META]) & tr
    res = {}
    for name, mask in (("all matches", tr), ("SFK-BP only (as 28 Sep)", sfk_only)):
        for seed in (0, 1, 2):
            netS = SC.train(X[mask], y[mask], epochs=2 if LOCAL else 20, batch=64, seed=seed, log=log); p = SC.score(netS, X)
            for sp in ("exam", "clip34"):
                idx = np.where(split == sp)[0]; M = [META[i] for i in idx]
                for fw in (0.0, 0.5, 1.0): res.setdefault(f"{name} | {sp} | scorer + {fw} x finder", []).append(list(map(int, SC.top1_per_frame(M, p[idx], fw))))
            if name == "all matches" and seed == 0: torch.save(netS.state_dict(), f"{WORK}/scorer_all_seed0.pt")
    for sp in ("exam", "clip34"):
        idx = np.where(split == sp)[0]; res[f"baseline | {sp} | finder score only"] = [list(map(int, SC.top1_per_frame([META[i] for i in idx], np.zeros(len(idx)), 1.0)))]
    for k, v in res.items(): log(f"{k:55s} " + "  ".join(f"{a}/{b}" for a, b in v))
    json.dump({"ok": True, "crops": int(len(X)), "train_ball": int(y[tr].sum()), "results": res, "minutes": round((time.time() - t0) / 60, 1)}, open(f"{WORK}/result.json", "w"), indent=1)
except Exception:
    import traceback; log(traceback.format_exc()); json.dump({"ok": False, "error": traceback.format_exc()[-4000:]}, open(f"{WORK}/result.json", "w"), indent=1)
