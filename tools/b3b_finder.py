"""B3b (3 Oct, free runner, CPU, no Modal): SoccerTrack v2 ball labels checked by our RF-DETR ball finder.

For every B3 label (results/free/b3/labels.json): read the video frame, let the app's finder (B7, ipanema/ballrf.py
weights) look at a 640-px window around the projection at 1x and 2x zoom (balls there are 3-10 px), keep the label only
when the finder is confident within 60 px of the projection and has no strong rival guess (ipanema/b3b.py).
Writes results/free/b3b/:
  hits.json     every label: the finder's guesses near the projection (both scales), keep / reason
  labels.json   kept labels: B3 fields + x, y moved to the finder's ball, conf
  crops64.npz   X = 64x64 around each kept ball
  kept_<m>.jpg  check sheet of kept balls (2x zoom, best first): green ring = finder, yellow tick = projection
  drop.jpg      random sample of dropped labels with the finder's best guess (red ring), to see what is missed
  summary.json  counts per match, per confidence
    HF_TOKEN=... python tools/b3b_finder.py
    DRY=1 ST_ROOT=<local stand-in> B3_LABELS=<labels.json> python tools/b3b_finder.py   (stand-in finder: bright blobs)
"""
import json, os, sys, time, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import b3b as BB

DRY = os.environ.get("DRY") == "1"
if not DRY:
    try: import rfdetr  # noqa
    except ImportError: subprocess.run("pip install -q rfdetr==1.11.0 torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple", shell=True, check=True)
from ipanema import soccertrack as ST

REPO = "atomscott/soccertrack-v2"; ROOT = os.environ.get("ST_ROOT")
LABELS = os.environ.get("B3_LABELS", "results/free/b3/labels.json")
OUT = os.environ.get("B3B_OUT", "results/free/b3b"); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("B3_TMP", "/tmp/b3"); os.makedirs(TMP, exist_ok=True)
SCALES = [float(s) for s in os.environ.get("B3B_SCALES", "1,2").split(",")]
MAX_MIN = float(os.environ.get("MAX_MIN", "300"))
t0 = time.time(); LOG = []
def log(m):
    LOG.append(f"{(time.time() - t0) / 60:6.1f} min  {m}"); print(LOG[-1], flush=True)
    open(f"{OUT}/log.txt", "w").write("\n".join(LOG) + "\n")

def fetch(rel):
    if ROOT: return os.path.join(ROOT, rel)
    from huggingface_hub import hf_hub_download
    t1 = time.time(); p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=TMP, token=os.environ.get("HF_TOKEN") or None)
    log(f"downloaded {rel}: {os.path.getsize(p) / 1e6:.0f} MB in {time.time() - t1:.0f} s"); return p

def drop(p):
    if not ROOT and p and os.path.exists(p): os.remove(p)

def read_frames(path, idx):
    cap = cv2.VideoCapture(path); out = {}; pos = -10 ** 9
    for k in sorted(set(idx)):
        if k < pos or k - pos > 100: cap.set(cv2.CAP_PROP_POS_FRAMES, k); pos = k
        while pos < k: cap.grab(); pos += 1
        ok, f = cap.read(); pos += 1
        if ok: out[k] = f
    cap.release(); return out


class Finder:
    """the app's RF-DETR ball finder on 640x640 RGB tiles -> [(x, y, conf)] per tile"""
    def __init__(self):
        from ipanema import ballrf
        self.m = ballrf.load(ballrf.WEIGHTS)
    def __call__(self, tiles_bgr, floor=0.05):
        res = self.m.predict([np.ascontiguousarray(t[:, :, ::-1]) for t in tiles_bgr], threshold=floor)
        if not isinstance(res, list): res = [res]
        return [[(float((a + c) / 2), float((b + e) / 2), float(cf)) for (a, b, c, e), cf in zip(d.xyxy, d.confidence)] for d in res]


class StandIn:
    """dry run: every small bright blob is a 'ball', conf from its brightness"""
    def __call__(self, tiles_bgr, floor=0.05):
        out = []
        for t in tiles_bgr:
            g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY); n, _, st, cen = cv2.connectedComponentsWithStats((g > 200).astype(np.uint8))
            out.append([(float(cen[i][0]), float(cen[i][1]), min(1.0, float(g[int(cen[i][1]), int(cen[i][0])]) / 255))
                        for i in range(1, n) if st[i, 4] < 400])
        return out


def tile(img, xy, proj, ring, z=2, half=40, txt=""):
    c = ST.crop(img, xy, 2 * half); c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST); o = half * z
    rx, ry = (proj[0] - xy[0]) * z + o, (proj[1] - xy[1]) * z + o
    cv2.line(c, (int(rx) - 9, int(ry) - 9), (int(rx) - 4, int(ry) - 4), (0, 255, 255), 2)
    if ring: cv2.circle(c, (o, o), 14, ring, 1)
    cv2.putText(c, txt, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return c

def sheet(tiles, path, cols=8):
    if not tiles: return
    h, w = tiles[0].shape[:2]; rows = (len(tiles) + cols - 1) // cols
    s = np.zeros((rows * (h + 2), cols * (w + 2), 3), np.uint8)
    for i, t in enumerate(tiles): r, c = divmod(i, cols); s[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = t
    cv2.imwrite(path, s, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    labels = json.load(open(LABELS)); finder = StandIn() if DRY else Finder(); log(f"{len(labels)} labels, scales {SCALES}, {'stand-in' if DRY else 'RF-DETR B7'}")
    by_vid = {}
    for i, l in enumerate(labels): by_vid.setdefault(l["video"], []).append(i)
    rows = [None] * len(labels); kept_tiles = {}; drop_tiles = []; X = []; rng = np.random.default_rng(7)
    for vid, idx in by_vid.items():
        if (time.time() - t0) / 60 > MAX_MIN: log("time limit"); break
        try: vp = fetch(vid)
        except Exception as e: log(f"{vid}: {e!r}"); continue
        imgs = read_frames(vp, [labels[i]["frame"] for i in idx]); t1 = time.time()
        for i in idx:
            l = labels[i]; img = imgs.get(l["frame"])
            if img is None: rows[i] = dict(i=i, keep=False, reason="frame not read"); continue
            proj = (l["x_raw"], l["y_raw"]); per = {}
            cuts = [BB.window(img, proj[0], proj[1], BB.SIZE, s) for s in SCALES]
            for s, (c, t), d in zip(SCALES, cuts, finder([c for c, _ in cuts])): per[s] = BB.near(BB.to_frame(d, t), proj)
            g = BB.merge_scales(per); keep, why = BB.decide(g)
            rows[i] = dict(i=i, match=l["match"], half=l["half"], frame=l["frame"], keep=keep, reason=why, guesses=g[:5],
                           dist_proj=round(float(np.hypot(g[0]["x"] - proj[0], g[0]["y"] - proj[1])), 1) if g else None,
                           b3_snapped=l["snapped"], ball_px=l["ball_px"], kind=l["kind"])
            if keep:
                b = g[0]; X.append(ST.crop(img, (b["x"], b["y"]), 64))
                kept_tiles.setdefault(l["match"], []).append((b["conf"], tile(img, (b["x"], b["y"]), proj, (0, 255, 0), txt=f"{b['conf']:.2f} {''.join('%g' % s for s in b['scales'])}")))
            elif rng.random() < 0.15:
                xy = (g[0]["x"], g[0]["y"]) if g else proj
                drop_tiles.append(tile(img, xy, proj, (0, 0, 255) if g else None, txt=f"{why[:4]} {g[0]['conf']:.2f}" if g else why[:8]))
        drop(vp); n = sum(1 for i in idx if rows[i] and rows[i]["keep"])
        log(f"{vid}: {len(idx)} labels, kept {n}, finder {time.time() - t1:.0f} s")
        json.dump([r for r in rows if r], open(f"{OUT}/hits.json", "w"), indent=0)
    done = [r for r in rows if r]; kept = [r for r in done if r["keep"]]
    for m, ts in kept_tiles.items(): sheet([t for _, t in sorted(ts, key=lambda z: -z[0])], f"{OUT}/kept_{m}.jpg")
    sheet(drop_tiles[:96], f"{OUT}/drop.jpg")
    out = []
    for r in kept:
        l = dict(labels[r["i"]]); b = r["guesses"][0]
        l.update(x=round(b["x"], 1), y=round(b["y"], 1), conf=round(b["conf"], 3), finder_scales=b["scales"], source="b3b_rfdetr"); out.append(l)
    json.dump(out, open(f"{OUT}/labels.json", "w"), indent=0)
    if X: np.savez_compressed(f"{OUT}/crops64.npz", X=np.array(X, np.uint8))
    best = [r["guesses"][0]["conf"] for r in done if r.get("guesses")]
    summ = dict(labels=len(labels), checked=len(done), kept=len(kept), minutes=round((time.time() - t0) / 60, 1), scales=SCALES,
                rule=dict(radius=BB.RADIUS, conf=BB.CONF, rival=BB.RIVAL),
                reasons={k: sum(r["reason"] == k for r in done) for k in sorted({r["reason"] for r in done})},
                best_conf_at_least={str(c): sum(b >= c for b in best) for c in (0.1, 0.3, 0.5, 0.7, 0.9)},
                kept_by_scale={",".join("%g" % s for s in sc): sum(r["guesses"][0]["scales"] == list(sc) for r in kept)
                               for sc in {tuple(r["guesses"][0]["scales"]) for r in kept}},
                kept_where_b3_snapped=sum(r["b3_snapped"] for r in kept),
                per_match={m: dict(labels=sum(r["match"] == m for r in done if "match" in r), kept=sum(r["match"] == m for r in kept))
                           for m in sorted({r["match"] for r in done if "match" in r})})
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1); log(f"done: kept {len(kept)}/{len(done)}  {summ['reasons']}")

if __name__ == "__main__":
    main()
