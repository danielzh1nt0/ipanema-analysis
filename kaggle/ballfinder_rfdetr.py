# Kaggle (free T4), 29 Sep: a NEW ball finder = RF-DETR (Apache-2.0, rfdetr==1.11.0) fine-tuned with one class "ball"
# on our approved labels; graded against the current click finder on the SFK-BP exam (108 frames) and the 34 clip moments.
# Training data: 640x640 crops around approved labels (conf >= 0.3, run >= 4) of Vasalund, Solheim, Spanga, Djursholm
# + Daniel's SFK-BP TRAIN clicks (never exam frames, never the clip's time window), ball off-centre at random,
# box 18 px (14 px in the far top 40% of the frame), plus ~20% ball-free crops.
# Grading: full 1920x1080 frame cut into 4x2 overlapping 640 px tiles (native scale = training scale), guesses >= 0.05.
#   exam, same definitions as results/ball/11/summary.json (BC.grade): a ball frame is right when one of the 3 best
#   guesses >= CONF is within 30 px; a no-ball frame is right when there is no guess >= CONF (CONF 0.25 as for the click
#   finder, plus a sweep because RF-DETR scores are on another scale); and as results/ball/probe.json: ball among
#   guesses >= 0.05 / top guess is the ball.  clip34: ball among guesses / top guess (as probe.json).
# MODE=smoke: 50 crops, 1 epoch, 3 exam frames.  MODE=full: the real run.
# LOCAL=1: offline dry run with stand-in files (R2_BASE = a local folder, REPO_DIR = this repo).
import os, pickle, sys, json, time, shutil, subprocess, random, urllib.request, numpy as np
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
R2 = os.environ.get("R2_BASE", "{{R2}}").rstrip("/"); LOCAL = os.environ.get("LOCAL") == "1"; MODE = os.environ.get("MODE", "full")
WORK = os.environ.get("WORK", "/kaggle/working"); TMP = os.environ.get("TMPD", "/kaggle/temp"); os.makedirs(WORK, exist_ok=True); os.makedirs(TMP, exist_ok=True)
SMOKE = MODE == "smoke"
SIZE = os.environ.get("SIZE", "small")                                          # small | medium | base
PER_MATCH = int(os.environ.get("PER_MATCH", "12" if SMOKE else "1150"))        # ball crops per training match
SFK_COPIES = int(os.environ.get("SFK_COPIES", "1" if SMOKE else "3"))           # SFK-BP is the exam ground: 3 crops per click
EPOCHS = int(os.environ.get("EPOCHS", "1" if SMOKE else "12")); BATCH = int(os.environ.get("BATCH", "2" if LOCAL else "4"))   # 8 ran out of T4 memory (29 Sep)
MAX_TRAIN_H = float(os.environ.get("MAX_TRAIN_H", "0.25" if SMOKE else "2.6"))   # hard stop for training (PTL max_time)
RES = int(os.environ.get("RES", "640")); CROP = 640; NEG_FRAC = 0.25            # 1 ball-free crop per 4 ball crops = 20% of all
MATCHES = ["p15u-vs-vasalund-2026-09-20", "p09-norrviken-vs-solheim-2026-08-30", "p15u-vs-spanga-2026-09-25", "p15u-vs-djursholm-2026-09-26"]
t0 = time.time(); LOG = []; REPORT = {"started": time.strftime("%Y-%m-%d %H:%M"), "mode": MODE, "size": SIZE, "res": RES, "epochs": EPOCHS, "batch": BATCH}
def log(m):
    LOG.append(f"{(time.time() - t0) / 60:6.1f} min  {m}"); print(LOG[-1], flush=True); open(f"{WORK}/log.txt", "w").write("\n".join(LOG) + "\n")
def save(): REPORT["minutes"] = round((time.time() - t0) / 60, 1); json.dump(REPORT, open(f"{WORK}/result.json", "w"), indent=1)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); log(f"$ {c[:120]} -> {r.returncode} {(r.stdout + r.stderr)[-300:].strip()}"); return r.returncode
def get(rel, dst):
    """R2 object -> local file (browser user agent: R2 answers 403 to Python's default one)"""
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); t1 = time.time()
    if not R2.startswith("http"): shutil.copy(f"{R2}/{rel}", dst); return dst
    url = f"{R2}/{rel}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ipanema-kaggle"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f: shutil.copyfileobj(r, f, 16 << 20)
    except Exception as e:
        log(f"urllib failed on {rel} ({e!r}); trying curl")
        if subprocess.run(["curl", "-fsSL", "--retry", "3", "-A", "Mozilla/5.0", "-o", dst + ".part", url]).returncode: raise RuntimeError(f"cannot download {rel}")
    os.replace(dst + ".part", dst); log(f"downloaded {rel}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t1:.0f} s"); return dst

def video(rel, dst):
    """local copy of a video (fast seeking); smoke runs and failed downloads read the URL directly (OpenCV can)"""
    if not R2.startswith("http"): return get(rel, dst)
    if SMOKE and "vasalund" not in rel: return f"{R2}/{rel}"
    try: return get(rel, dst)
    except Exception as e: log(f"download failed ({e!r}); reading {rel} from the URL"); return f"{R2}/{rel}"
def rm(p):
    if os.path.exists(p): os.remove(p)
ball = None
def frames_at(path, wanted):
    """yield (frame_index, BGR image) for the sorted wanted frame indices: grab forward when close, seek when far"""
    import cv2
    cap = cv2.VideoCapture(path); pos = 0
    for k in sorted(set(int(w) for w in wanted)):
        if k < pos or k - pos > 250: cap.set(cv2.CAP_PROP_POS_FRAMES, k); pos = k
        while pos < k:
            if not cap.grab(): break
            pos += 1
        ok, f = cap.read(); pos += 1
        if ok and pos - 1 == k: yield k, f
    cap.release()

try:
    REPO = os.environ.get("REPO_DIR", f"{TMP}/ia")
    if not LOCAL:
        sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        sh(f"rm -rf {REPO} && git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {REPO}")
        sh('pip install -q "rfdetr[train]==1.11.0"')
    sys.path.insert(0, REPO)
    import cv2, torch, rfdetr
    REPORT["versions"] = {"torch": torch.__version__, "rfdetr": getattr(rfdetr, "__version__", "?"), "cuda": torch.cuda.is_available()}; log(f"versions {REPORT['versions']}")
    rng = random.Random(0)

    # ---------------- dataset (YOLO layout, which rfdetr 1.11 reads: data.yaml + train/ valid/ with images + labels)
    DS = f"{TMP}/ds"; shutil.rmtree(DS, ignore_errors=True)
    for s in ("train", "valid"):
        for sub in ("images", "labels"): os.makedirs(f"{DS}/{s}/{sub}", exist_ok=True)
    open(f"{DS}/data.yaml", "w").write(f"path: {DS}\ntrain: train/images\nval: valid/images\nnc: 1\nnames: ['ball']\n")
    counts = {"train_ball": 0, "train_empty": 0, "valid_ball": 0, "valid_empty": 0}
    def box_px(y, h): return 14.0 if y < 0.4 * h else 18.0                       # far balls are smaller
    def put(split, name, f, x, y):
        """one crop of frame f; (x, y) = ball or None for a ball-free crop"""
        h, w = f.shape[:2]
        if x is not None:
            x0 = int(min(max(0, x - rng.randint(CROP // 8, CROP * 7 // 8)), w - CROP)); y0 = int(min(max(0, y - rng.randint(CROP // 8, CROP * 7 // 8)), h - CROP))
        else:
            for _ in range(30):
                x0, y0 = rng.randint(0, w - CROP), rng.randint(0, h - CROP)
                if ball is None or not (x0 - 40 <= ball[0] <= x0 + CROP + 40 and y0 - 40 <= ball[1] <= y0 + CROP + 40): break
            else: return
        cv2.imwrite(f"{DS}/{split}/images/{name}.jpg", f[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(f"{DS}/{split}/labels/{name}.txt", "w") as fh:
            if x is not None: b = box_px(y, h); fh.write(f"0 {(x - x0) / CROP:.6f} {(y - y0) / CROP:.6f} {b / CROP:.6f} {b / CROP:.6f}\n")
        counts[f"{split}_{'ball' if x is not None else 'empty'}"] += 1
    def put_box(split, name, f, x, y, bw, bh):
        """a ball crop with the picture's own box size (outside sets)"""
        h, w = f.shape[:2]
        x0 = int(min(max(0, x - rng.randint(CROP // 8, CROP * 7 // 8)), w - CROP)); y0 = int(min(max(0, y - rng.randint(CROP // 8, CROP * 7 // 8)), h - CROP))
        cv2.imwrite(f"{DS}/{split}/images/{name}.jpg", f[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{DS}/{split}/labels/{name}.txt", "w").write(f"0 {(x - x0) / CROP:.6f} {(y - y0) / CROP:.6f} {bw / CROP:.6f} {bh / CROP:.6f}\n")
        counts[f"{split}_ball"] += 1
    # 2 Oct (S8, kaggle/ballfinder_feet.py): AT_FEET=1 = balls at a player's feet / half hidden are the finder's weak spot
    # (dead-ball stretch: ball at a foot scores 0.11, a line mark 0.19). Every labelled ball that sits in the lower half of a
    # person box (or just under it) gets FEET_COPIES extra crops, cut so the player is in the crop. Person boxes from the
    # pipeline's own detector (RF-DETR medium) on the labelled frame. Test clip and exam frames never trained on (unchanged).
    AT_FEET = os.environ.get("AT_FEET") == "1"; FEET_COPIES = int(os.environ.get("FEET_COPIES", "3")); REPORT["at_feet_on"] = AT_FEET
    feetc = {"checked": 0, "at_feet": 0, "copies": 0}
    if AT_FEET:
        os.environ["IPANEMA_DETECTOR"] = "rfdetr"; from ipanema import tracking as TR
        if LOCAL:                                                                 # dry run: a stand-in detector (one box with the stub ball at its feet)
            class _Stub:
                def detect_batch(self, fs, conf, tiles):
                    class D: xyxy = np.array([[260.0 + 5 * 0, 520.0, 340.0 + 5 * 150, 612.0]])
                    return [[D()]]
            pdet = _Stub()
        else: pdet = TR.RFDetrPerson("medium")
    def at_feet(f, x, y):
        if not AT_FEET or x is None: return False
        feetc["checked"] += 1
        try: d = pdet.detect_batch([f], 0.3, TR.FOLLOW_TILES)[0][0]
        except Exception: return False
        for (a, b, c, e) in d.xyxy:
            if a - 10 <= x <= c + 10 and (b + e) / 2 <= y <= e + 12: feetc["at_feet"] += 1; return True
        return False
    def put_feet(split, name, f, x, y):
        """FEET_COPIES crops that keep the player: ball offset by at most a third of the crop"""
        h, w = f.shape[:2]
        for j in range(FEET_COPIES):
            x0 = int(min(max(0, x - rng.randint(CROP // 3, CROP * 2 // 3)), w - CROP)); y0 = int(min(max(0, y - rng.randint(CROP // 3, CROP * 2 // 3)), h - CROP))
            cv2.imwrite(f"{DS}/{split}/images/{name}_ft{j}.jpg", f[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
            b = box_px(y, h); open(f"{DS}/{split}/labels/{name}_ft{j}.txt", "w").write(f"0 {(x - x0) / CROP:.6f} {(y - y0) / CROP:.6f} {b / CROP:.6f} {b / CROP:.6f}\n")
            counts[f"{split}_ball"] += 1; feetc["copies"] += 1
    REPORT["matches"] = {}
    for mt in MATCHES:
        try:
            labs = json.load(open(get(f"labels/{mt}_trainset_clicks.json", f"{TMP}/{mt}_labels.json")))["labels"]
            good = sorted([l for l in labs if l["conf"] >= 0.3 and l.get("run", 0) >= 4], key=lambda l: l["frame"])
            n = min(PER_MATCH, len(good)); step = len(good) / max(n, 1)
            pick = [good[int(i * step)] for i in range(n)]                       # spread over the whole match
            nval = 0 if SMOKE else max(1, n // 25)                               # ~4% of frames held out for RF-DETR's own val
            vset = set(rng.sample(range(n), nval)) if nval else set()
            by = {int(l["frame"]): (i, l) for i, l in enumerate(pick)}
            vid = video(f"{mt}/video.mp4", f"{TMP}/{mt}.mp4"); c0 = dict(counts); t1 = time.time()
            for k, f in frames_at(vid, list(by)):
                i, l = by[k]; ball = (l["x"], l["y"]); split = "valid" if i in vset else "train"; nm = f"{mt[:12]}_{k:06d}"
                put(split, nm + "_b", f, l["x"], l["y"])
                if split == "train" and at_feet(f, l["x"], l["y"]): put_feet("train", nm, f, l["x"], l["y"])
                if rng.random() < NEG_FRAC: put(split, nm + "_n", f, None, None)
            rm(vid)
            REPORT["matches"][mt] = {"labels": len(labs), "usable": len(good), "frames_used": n, "crops": {k: counts[k] - c0[k] for k in counts}, "sec": round(time.time() - t1)}
            log(f"{mt}: {len(labs)} labels, {len(good)} usable, {n} frames -> {REPORT['matches'][mt]['crops']} in {time.time() - t1:.0f}s"); save()
        except Exception as e:
            log(f"{mt}: skipped ({e!r})"); REPORT["matches"][mt] = {"error": repr(e)[:300]}
    # SFK-BP: train clicks only; never exam frames (+-1 s) and never inside the 34-moment clip's window
    clicks = json.load(open(os.environ.get("CLICKS", f"{REPO}/results/labels/SFKBP1109_ball_clicks.json")))["frames"]
    exam = [r for r in clicks if r["split"] == "exam"]; exam_t = [r["t"] for r in exam]
    clip = video("SFKBP1109_s1200/video.mp4", f"{TMP}/clip.mp4"); cc = cv2.VideoCapture(clip); clip_fps = cc.get(cv2.CAP_PROP_FPS) or 29.97; clip_n = int(cc.get(cv2.CAP_PROP_FRAME_COUNT)); cc.release()
    clip_win = (1200 - 5, 1200 + clip_n / clip_fps + 5)                         # clip = full match from 1200 s
    train = [r for r in clicks if r["split"] == "train" and min(abs(r["t"] - t) for t in exam_t) > 1.0 and not (clip_win[0] <= r["t"] <= clip_win[1])]
    if SMOKE: train = [r for r in train if r["x"] is not None][:6]
    REPORT["sfk"] = {"train_clicks": sum(r["split"] == "train" for r in clicks), "used": len(train), "clip_window_s": [round(clip_win[0]), round(clip_win[1])], "exam": len(exam)}
    full = video("SFKBP1109/video.mp4", f"{TMP}/sfk_full.mp4"); cf = cv2.VideoCapture(full); fps = cf.get(cv2.CAP_PROP_FPS) or 29.97; cf.release()
    by = {int(round(r["t"] * fps)): r for r in train}; c0 = dict(counts)
    for k, f in frames_at(full, list(by)):
        r = by[k]; nm = f"sfk_{r['file'][:-4]}"
        if r["x"] is None: ball = None; put("train", nm + "_n", f, None, None)
        else:
            ball = (r["x"], r["y"])
            for j in range(SFK_COPIES): put("train", f"{nm}_b{j}", f, r["x"], r["y"])
            if at_feet(f, r["x"], r["y"]): put_feet("train", nm, f, r["x"], r["y"])
            if rng.random() < NEG_FRAC: put("train", nm + "_n", f, None, None)
    REPORT["sfk"]["crops"] = {k: counts[k] - c0[k] for k in counts}
    if AT_FEET: REPORT["at_feet"] = feetc; log(f"at feet: {feetc}"); del pdet; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # H1b (1 Oct, Daniel: yes, testing only): HF_EXTRA=1 adds the outside Bundesliga TV ball set (martinjolif/football-ball-detection,
    # 1,237 pictures, licence doubtful -> the weights from such a run are for testing, never promoted). One crop per picture with
    # its own box size (median 12 px, Veo-sized), all to train (their test split is never our exam), plus the usual ball-free share.
    HF_EXTRA = os.environ.get("HF_EXTRA") == "1"; REPORT["hf_extra"] = HF_EXTRA
    if HF_EXTRA:
        from ipanema import hfball
        if os.environ.get("HF_ROOT"): hroot = os.environ["HF_ROOT"]
        else:
            sh("pip install -q huggingface_hub")
            from huggingface_hub import snapshot_download
            hroot = snapshot_download(repo_id="martinjolif/football-ball-detection", repo_type="dataset", local_dir=f"{TMP}/hf_ball")
        layout, hitems, hnames = hfball.load(hroot); c0 = dict(counts); nused = 0
        if SMOKE: hitems = hitems[:6]
        for i, it in enumerate(hitems):
            im = hfball.image(it)
            if im is None or im.shape[0] < CROP or im.shape[1] < CROP: continue
            bx = hfball.balls(it, im, hnames)
            if len(bx) != 1: continue
            x, y, bw, bh = bx[0]; cx, cy = x + bw / 2, y + bh / 2; ball = (cx, cy); nm = f"hf_{i:05d}"
            put_box("train", nm + "_b", im, cx, cy, max(bw, 4.0), max(bh, 4.0)); nused += 1
            if rng.random() < NEG_FRAC: put("train", nm + "_n", im, None, None)
        REPORT["hf"] = {"layout": layout, "items": len(hitems), "used": nused, "crops": {k: counts[k] - c0[k] for k in counts}}
        log(f"HF extra: {REPORT['hf']}")
    if counts["valid_ball"] == 0:                                                 # smoke: RF-DETR needs a val split; copy 3 train crops
        for fn in sorted(os.listdir(f"{DS}/train/images"))[:3]:
            shutil.copy(f"{DS}/train/images/{fn}", f"{DS}/valid/images/{fn}"); shutil.copy(f"{DS}/train/labels/{fn[:-4]}.txt", f"{DS}/valid/labels/{fn[:-4]}.txt"); counts["valid_ball"] += 1
    if SMOKE:                                                                     # smoke: exactly ~50 training crops
        ims = sorted(os.listdir(f"{DS}/train/images"))
        for fn in ims[50:]: os.remove(f"{DS}/train/images/{fn}"); os.remove(f"{DS}/train/labels/{fn[:-4]}.txt")
        counts["train_total_after_cut"] = min(50, len(ims))
    REPORT["crops"] = counts; log(f"dataset: {counts}"); save()

    # 1 Oct (fix ball): HARD_NEG=1 = hard-example oversampling. The remaining app misses are mostly the finder scoring a shoe, a
    # player or a second ball higher than the match ball (results/qa/ballmiss2). The current finder (B7) looks at every training
    # crop; crops where it has a confident guess (>= HN_CONF) away from the labelled ball are copied HN_COPIES more times, so
    # training sees those look-alikes more often (unlabelled = background for RF-DETR). Clip and exam frames are never in training.
    HARD_NEG = os.environ.get("HARD_NEG") == "1"; REPORT["hard_neg_on"] = HARD_NEG
    if HARD_NEG:
        from ipanema import ballrf
        HN_CONF = float(os.environ.get("HN_CONF", "0.4")); HN_COPIES = int(os.environ.get("HN_COPIES", "2"))
        old = ballrf.load(f"{REPO}/{ballrf.WEIGHTS}") if not os.environ.get("HN_WEIGHTS") else ballrf.load(os.environ["HN_WEIGHTS"])
        ims = sorted(os.listdir(f"{DS}/train/images")); hard = []; t1 = time.time()
        for b0 in range(0, len(ims), 16):
            chunk = ims[b0:b0 + 16]; imgs = [cv2.imread(f"{DS}/train/images/{fn}")[:, :, ::-1].copy() for fn in chunk]
            res = old.predict(imgs, threshold=HN_CONF)
            if not isinstance(res, list): res = [res]
            for fn, d in zip(chunk, res):
                lab = open(f"{DS}/train/labels/{fn[:-4]}.txt").read().split()
                bx = (float(lab[1]) * CROP, float(lab[2]) * CROP) if lab else None
                fp = [float(cf) for (a, b, c, e), cf in zip(d.xyxy, d.confidence) if bx is None or np.hypot((a + c) / 2 - bx[0], (b + e) / 2 - bx[1]) > 20]
                if fp: hard.append((fn, max(fp), bx is not None))
        for fn, _, _ in hard:
            for j in range(HN_COPIES):
                shutil.copy(f"{DS}/train/images/{fn}", f"{DS}/train/images/{fn[:-4]}_hn{j}.jpg"); shutil.copy(f"{DS}/train/labels/{fn[:-4]}.txt", f"{DS}/train/labels/{fn[:-4]}_hn{j}.txt")
        REPORT["hard_neg"] = {"conf": HN_CONF, "copies": HN_COPIES, "crops_checked": len(ims), "hard_crops": len(hard), "hard_with_ball": sum(h[2] for h in hard),
                              "hard_empty": sum(not h[2] for h in hard), "sec": round(time.time() - t1)}
        log(f"hard negatives: {REPORT['hard_neg']}"); del old; torch.cuda.empty_cache() if torch.cuda.is_available() else None; save()

    # 2 Oct (S8, kaggle/ballfinder_neg.py): explicit 'not a ball' crops. NEG_JSON = repo file of {clip, frame, x, y} spots (mined from
    # the exact clip inputs: candidates away from the keyed ball, still spots with nobody near); NEG_CLIPS = which clips of it to
    # use (the SFK-BP clip is the test set and is never used). NEG_SFK_FP=1 = the old finder on SFK-BP TRAIN click frames (outside
    # the exam and the clip window): every confident guess away from the click becomes an empty crop. NEG_COPIES copies each.
    NEG_JSON = os.environ.get("NEG_JSON"); NEG_SFK_FP = os.environ.get("NEG_SFK_FP") == "1"; NEG_COPIES = int(os.environ.get("NEG_COPIES", "2"))
    if NEG_JSON or NEG_SFK_FP:
        from ipanema import ballrf
        t1 = time.time(); negc = {"json": 0, "sfk_fp": 0, "sfk_frames": 0}
        def put_empty_at(name, f, x, y, ball_xy=None):
            """a crop around a 'not a ball' spot; the real ball (if known and inside the crop) keeps its box, the spot gets none"""
            h, w = f.shape[:2]
            x0 = int(min(max(0, x - rng.randint(CROP // 8, CROP * 7 // 8)), w - CROP)); y0 = int(min(max(0, y - rng.randint(CROP // 8, CROP * 7 // 8)), h - CROP))
            inside = ball_xy is not None and x0 + 6 <= ball_xy[0] <= x0 + CROP - 6 and y0 + 6 <= ball_xy[1] <= y0 + CROP - 6
            for j in range(NEG_COPIES):
                cv2.imwrite(f"{DS}/train/images/{name}_n{j}.jpg", f[y0:y0 + CROP, x0:x0 + CROP], [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(f"{DS}/train/labels/{name}_n{j}.txt", "w") as fh:
                    if inside: b = box_px(ball_xy[1], h); fh.write(f"0 {(ball_xy[0] - x0) / CROP:.6f} {(ball_xy[1] - y0) / CROP:.6f} {b / CROP:.6f} {b / CROP:.6f}\n")
                counts["train_ball" if inside else "train_empty"] += 1
        if NEG_JSON:
            spots = json.load(open(f"{REPO}/{NEG_JSON}")); use = [c for c in os.environ.get("NEG_CLIPS", "p15u-vs-aik-2026-09-21-bd09_s2520").split(",") if c]
            per_clip = int(os.environ.get("NEG_PER_CLIP", "400")); rng2 = random.Random(1)
            for c in use:
                rows = [r for r in spots if r["clip"] == c]; rng2.shuffle(rows); rows = rows[:per_clip]; byf = {}
                for r in rows: byf.setdefault(r["frame"], []).append(r)
                vp = video(f"{c}/video.mp4", f"{TMP}/{c}_neg.mp4")
                for k, f in frames_at(vp, list(byf)):
                    f = cv2.resize(f, (1920, 1080))
                    for i, r in enumerate(byf[k]): put_empty_at(f"neg_{c[:6]}_{k}_{i}", f, r["x"], r["y"], r.get("ball")); negc["json"] += 1
                if vp.startswith(TMP): rm(vp)
        if NEG_SFK_FP:
            old = ballrf.load(f"{REPO}/{ballrf.WEIGHTS}"); FP_CONF = float(os.environ.get("NEG_FP_CONF", "0.4")); pick = train[::max(1, len(train) // int(os.environ.get("NEG_SFK_FRAMES", "600")))]
            byk2 = {int(round(r["t"] * fps)): r for r in pick}; buf = []
            def flush():
                if not buf: return
                res = ballrf.detect_many(old, [f for _, f in buf], floor=FP_CONF)
                for (k, f), g in zip(buf, res):
                    r = byk2[k]; negc["sfk_frames"] += 1
                    for i, (x, y, cf) in enumerate(g):
                        if r["x"] is None or np.hypot(x - r["x"], y - r["y"]) > 40: put_empty_at(f"negfp_sfk_{k}_{i}", f, x, y, None if r["x"] is None else (r["x"], r["y"])); negc["sfk_fp"] += 1
                buf.clear()
            for k, f in frames_at(full, list(byk2)):
                buf.append((k, cv2.resize(f, (1920, 1080))))
                if len(buf) >= 4: flush()
            flush(); del old; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        REPORT["negatives"] = {**negc, "copies": NEG_COPIES, "sec": round(time.time() - t1)}; REPORT["crops"] = counts; log(f"negatives: {REPORT['negatives']} -> {counts}"); save()

    # ---------------- train
    import rfdetr.training as RT
    _bt = RT.build_trainer
    def _bt_capped(*a, **k):                                                      # hard time cap on training (PTL max_time)
        k.setdefault("max_time", {"hours": MAX_TRAIN_H}); return _bt(*a, **k)
    RT.build_trainer = _bt_capped
    Model = {"small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase}[SIZE]
    model = Model(pretrain_weights=None) if os.environ.get("PRETRAIN") == "none" else Model(); OUTD = f"{TMP}/rf_out"; t1 = time.time(); log(f"training RF-DETR {SIZE} at {RES} px, {EPOCHS} epochs, batch {BATCH}, cap {MAX_TRAIN_H} h")
    kw = dict(dataset_dir=DS, dataset_file="yolo", epochs=EPOCHS, batch_size=BATCH, grad_accum_steps=max(1, 16 // BATCH), resolution=RES, output_dir=OUTD,
              num_workers=0 if LOCAL else 4, expanded_scales=False, tensorboard=False, progress_bar=None, checkpoint_interval=100, seed=0, lr_scheduler=os.environ.get("SCHED", "cosine"))
    if not LOCAL: kw["amp_dtype"] = "fp16"                                       # T4 has no real bf16 ("auto" picked bf16 = slow + more memory)
    if LOCAL: kw["device"] = "cpu"
    model.train(**kw)
    REPORT["train_min"] = round((time.time() - t1) / 60, 1); REPORT["train_images"] = len(os.listdir(f"{DS}/train/images")); REPORT["train_img_per_s"] = round(EPOCHS * REPORT["train_images"] / max(1, time.time() - t1), 2)
    log(f"training done in {REPORT['train_min']} min; outputs {sorted(os.listdir(OUTD))}"); save()
    for fn in ("metrics.csv", "training_config.json"):
        if os.path.exists(f"{OUTD}/{fn}"): shutil.copy(f"{OUTD}/{fn}", f"{WORK}/{fn}")
    ck = next((f"{OUTD}/{c}" for c in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth") if os.path.exists(f"{OUTD}/{c}")), None)
    if ck:
        model = Model.from_checkpoint(ck, resolution=RES); REPORT["checkpoint"] = os.path.basename(ck); log(f"loaded {ck}: resolution {getattr(getattr(model, 'model_config', None), 'resolution', '?')}")
        sd = torch.load(ck, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
        sd = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}
        wname = f"rfdetr_ball_{SIZE}_{time.strftime('%Y%m%d')}{'_hf_TESTONLY' if HF_EXTRA else ''}{'_hn' if HARD_NEG else ''}_fp16.pth"; torch.save({"model": sd, "size": SIZE, "resolution": RES, "classes": ["ball"]}, f"{WORK}/{wname}")
        REPORT["weights"] = {"file": wname, "MB": round(os.path.getsize(f"{WORK}/{wname}") / 1e6, 1), "full_checkpoint_MB": round(os.path.getsize(ck) / 1e6, 1)}
        if REPORT["weights"]["MB"] > 95: os.rename(f"{WORK}/{wname}", f"{TMP}/{wname}"); REPORT["weights"]["note"] = "too big for git: left on Kaggle temp (lost)"
    log(f"weights {REPORT.get('weights')}"); save()

    # ---------------- tiled full-frame detection
    TX, TY = [0, 427, 853, 1280], [0, 440]                                         # 4x2 tiles of 640 on 1920x1080, overlapping
    def detect(img, floor=0.05):
        h, w = img.shape[:2]; xs = [min(x, w - CROP) for x in TX]; ys = [min(y, h - CROP) for y in TY]
        tiles = [(x, y) for y in ys for x in xs]; rgb = img[:, :, ::-1]
        res = model.predict([np.ascontiguousarray(rgb[y:y + CROP, x:x + CROP]) for x, y in tiles], threshold=floor)
        if not isinstance(res, list): res = [res]
        det = []
        for (x, y), d in zip(tiles, res):
            for (a, b, c, e), cf in zip(d.xyxy, d.confidence): det.append((float(x + (a + c) / 2), float(y + (b + e) / 2), float(cf)))
        det.sort(key=lambda z: -z[2]); keep = []
        for z in det:
            if all(np.hypot(z[0] - k[0], z[1] - k[1]) > 12 for k in keep): keep.append(z)
        return keep
    def near(g, t, px=30.0): return bool(t is not None and np.hypot(g[0] - t[0], g[1] - t[1]) <= px)

    # ---------------- exam (108 SFK-BP frames, full frame)
    ex = exam[:3] if SMOKE else exam; byk = {int(round(r["t"] * fps)): r for r in ex}; rows = []; t1 = time.time()
    for k, f in frames_at(full, list(byk)):
        r = byk[k]; g = detect(f); truth = (r["x"], r["y"]) if r["x"] is not None else None
        rows.append({"file": r["file"], "truth": truth, "guesses": [[round(a, 1), round(b, 1), round(c, 4)] for a, b, c in g[:20]]})
    log(f"exam: {len(rows)} frames graded in {time.time() - t1:.0f}s")
    def exam_score(conf):
        b = [r for r in rows if r["truth"]]; nb = [r for r in rows if not r["truth"]]
        found = sum(any(near(z, r["truth"]) for z in [z for z in r["guesses"] if z[2] >= conf][:3]) for r in b)
        quiet = sum(not any(z[2] >= conf for z in r["guesses"]) for r in nb)
        return {"correct": found + quiet, "of": len(rows), "ball_found": found, "ball_frames": len(b), "no_ball_right": quiet, "no_ball_frames": len(nb)}
    b = [r for r in rows if r["truth"]]
    REPORT["exam"] = {"at_0.25_as_click_finder": exam_score(0.25), "sweep": {str(c): exam_score(c) for c in (0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6)},
                      "ball_among_guesses_0.05": sum(any(near(z, r["truth"]) for z in r["guesses"]) for r in b),
                      "top_guess_is_ball": sum(bool(r["guesses"]) and near(r["guesses"][0], r["truth"]) for r in b), "ball_frames": len(b),
                      "current_click_finder": {"correct": 70, "ball_found": 47, "among_guesses": 54, "top_guess": 51}, "wasb": {"among_guesses": 79, "top_guess": 56}}
    REPORT["exam_rows"] = rows; log(f"exam: {json.dumps({k: v for k, v in REPORT['exam'].items() if k != 'sweep'})}"); save()

    # ---------------- 34 clip moments
    mom = json.load(open(os.environ.get("MOMENTS", f"{REPO}/results/picker/moments34.json"))); items = list(mom.items())[:3] if SMOKE else list(mom.items())
    byk = {int(k): v for k, v in items}; crow = []
    for k, f in frames_at(clip, list(byk)):
        g = detect(f); crow.append({"frame": k, "truth": byk[k]["truth"], "guesses": [[round(a, 1), round(b, 1), round(c, 4)] for a, b, c in g[:20]]})
    REPORT["clip34"] = {"moments": len(crow), "ball_among_guesses_0.05": sum(any(near(z, r["truth"]) for z in r["guesses"]) for r in crow),
                        "ball_among_top6": sum(any(near(z, r["truth"]) for z in r["guesses"][:6]) for r in crow),
                        "top_guess_is_ball": sum(bool(r["guesses"]) and near(r["guesses"][0], r["truth"]) for r in crow),
                        "current_click_finder": {"among_guesses": 18, "top_guess": 16}, "wasb": {"among_guesses": 31, "top_guess": 26}}
    REPORT["clip_rows"] = crow; log(f"clip34: {json.dumps(REPORT['clip34'])}")
    # 2 Oct (S8): candidates over whole clips with the new finder (and the old one, same code), for the offline picker/pass grading
    for c in [c for c in os.environ.get("CANDS_CLIPS", "").split(",") if c]:
        vp = video(f"{c}/video.mp4", f"{TMP}/{c}_cands.mp4"); t1 = time.time()
        newc = ballrf.candidates_model(model, vp, floor=0.05, batch=4, log=log) if hasattr(ballrf, "candidates_model") else None
        if newc is not None: pickle.dump(newc, open(f"{WORK}/cands_new_{c}.pkl", "wb"))
        old = ballrf.load(f"{REPO}/{ballrf.WEIGHTS}"); oldc = ballrf.candidates_model(old, vp, floor=0.05, batch=4, log=log); pickle.dump(oldc, open(f"{WORK}/cands_old_{c}.pkl", "wb")); del old
        log(f"candidates {c}: new {len(newc) if newc else None} old {len(oldc)} frames in {(time.time() - t1) / 60:.1f} min")
    REPORT["ok"] = True; save(); log("done")
except Exception:
    import traceback; log(traceback.format_exc()); REPORT["ok"] = False; REPORT["error"] = traceback.format_exc()[-3000:]; save()
