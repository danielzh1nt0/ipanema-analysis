"""B3 (3 Oct, free runner, no Modal): SoccerTrack v2 ball labels in pixels, ground balls only.

For each of the 10 matches: keypoints -> fisheye calibration (dataset's own chain, ipanema/soccertrack.py), tracker XML
-> ball per frame, ground balls only (at a player's feet or rolling), N_PER_HALF frames per half spread in time,
projected to the 4K panorama, snapped to the ball blob nearby. Writes to results/free/b3/:
  labels.json      every label: match, half, video frame, raw + snapped pixel position, ball size px, kind, ...
  crops64.npz      64x64 crops centred on each label (X) + one ball-free crop per label from the same frame (N)
  sheet_<m>.jpg    check sheet: 2x zoom around each label, yellow tick = projection, cyan ring = snapped ball
  over_<m>.jpg     one whole frame per match with all players (green) and the ball (yellow) projected
  summary.json     counts, calibration rms, snap rate per match; listing + samples of the dataset's ball/ folder
Training jobs (Kaggle, internet) cut their own 640 crops from the HF videos with labels.json.
LOCAL dry run: ST_ROOT=<folder laid out like the HF repo> python tools/b3_soccertrack.py
"""
import json, os, shutil, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import soccertrack as ST

REPO = "atomscott/soccertrack-v2"
ROOT = os.environ.get("ST_ROOT")                     # local stand-in (dry run); None = Hugging Face
OUT = os.environ.get("B3_OUT", "results/free/b3"); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get("B3_TMP", "/tmp/b3"); os.makedirs(TMP, exist_ok=True)
N_PER_HALF = int(os.environ.get("N_PER_HALF", "40")); MIN_GAP = int(os.environ.get("MIN_GAP", "250"))  # 10 s at 25 fps
MAX_MIN = float(os.environ.get("MAX_MIN", "300")); XML_FPS = 25.0; MIN_HIT = float(os.environ.get("MIN_HIT", "0.6"))
MATCHES = os.environ.get("ST_MATCHES", ",".join(ST.MATCHES)).split(",")
CORR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference/soccertrack")
t0 = time.time(); LOG = []
def log(m):
    LOG.append(f"{(time.time() - t0) / 60:6.1f} min  {m}"); print(LOG[-1], flush=True)
    open(f"{OUT}/log.txt", "w").write("\n".join(LOG) + "\n")

def listing():
    if ROOT: return sorted(os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) for f in fs)
    from huggingface_hub import HfApi
    return sorted(s.rfilename for s in HfApi(token=os.environ.get("HF_TOKEN") or None).dataset_info(REPO).siblings)

def fetch(rel):
    if ROOT: return os.path.join(ROOT, rel)
    from huggingface_hub import hf_hub_download
    t1 = time.time(); p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=TMP, token=os.environ.get("HF_TOKEN") or None)
    log(f"downloaded {rel}: {os.path.getsize(p) / 1e6:.0f} MB in {time.time() - t1:.0f} s"); return p

def drop(p):
    if not ROOT and p and os.path.exists(p): os.remove(p)

def find(files, m, *needles):
    c = [f for f in files if f"/{m}" in "/" + f and all(n in f for n in needles)]
    return sorted(c, key=len)[0] if c else None

def read_frames(path, idx):
    cap = cv2.VideoCapture(path); out = {}; pos = -10 ** 9
    for k in sorted(idx):
        if k < pos or k - pos > 100: cap.set(cv2.CAP_PROP_POS_FRAMES, k); pos = k
        while pos < k: cap.grab(); pos += 1
        ok, f = cap.read(); pos += 1
        if ok: out[k] = f
    cap.release(); return out

def tile(img, lab, z=2, half=40):
    c = ST.crop(img, (lab["x"], lab["y"]), 2 * half); c = cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    o = half * z; rx, ry = (lab["x_raw"] - lab["x"]) * z + o, (lab["y_raw"] - lab["y"]) * z + o
    cv2.line(c, (int(rx) - 9, int(ry) - 9), (int(rx) - 4, int(ry) - 4), (0, 255, 255), 2)        # projection: yellow tick
    if lab["snapped"]: cv2.circle(c, (o, o), int(max(8, lab["ball_px"] * z)), (255, 255, 0), 1)  # snapped: cyan ring
    cv2.putText(c, f"{lab['kind'][0]} {lab['snap_px']:.0f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return c

def sheet(tiles, path, cols=8):
    if not tiles: return
    h, w = tiles[0].shape[:2]; rows = (len(tiles) + cols - 1) // cols
    s = np.zeros((rows * (h + 2), cols * (w + 2), 3), np.uint8)
    for i, t in enumerate(tiles): r, c = divmod(i, cols); s[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = t
    cv2.imwrite(path, s, [cv2.IMWRITE_JPEG_QUALITY, 88])

def main():
    files = listing(); summ = {"started": time.strftime("%Y-%m-%d %H:%M"), "n_files": len(files), "matches": {}}
    summ["ball_folder"] = [f for f in files if f.startswith("ball/")]; summ["raw_folder"] = [f for f in files if f.startswith("raw/")]
    samples = {}
    for f in summ["ball_folder"][:3]:
        try: samples[f] = open(fetch(f), errors="replace").read()[:1200]
        except Exception as e: samples[f] = f"failed: {e!r}"
    summ["ball_folder_samples"] = samples
    labels, X, N = [], [], []
    for m in MATCHES:
        if (time.time() - t0) / 60 > MAX_MIN: log("time limit"); break
        ms = summ["matches"][m] = {}
        kp, xml = find(files, m, "keypoints", ".json"), find(files, m, "tracker_box_data", ".xml")
        vids = {h: find(files, m, f"panorama_{h}_half", ".mp4") for h in ("1st", "2nd")}
        ms.update(keypoints=kp, xml=xml, videos=vids)
        if not (kp and xml): log(f"{m}: missing keypoints/xml"); ms["status"] = "missing input"; continue
        try:
            pitch, image, src = ST.load_keypoints(fetch(kp), m, os.path.join(CORR, f"{m}_keypoints.json")); xp = fetch(xml)
            per = ST.parse_xml(xp); drop(xp)
        except Exception as e: log(f"{m}: {e!r}"); ms["status"] = f"error {e!r}"; continue
        ms["keypoint_source"] = src; cal = None; tiles = []
        for h, period in (("1st", "FIRST_HALF"), ("2nd", "SECOND_HALF")):
            p = per[period]; hs = ms[h] = {"xml_frames": int(len(p["frames"]))}
            if not len(p["frames"]) or not vids[h]: hs["status"] = "no data"; continue
            off = int(p["frames"][0]); g = ST.ground_candidates(p, m, XML_FPS)
            hs.update(offset=off, ball_tracked=int(np.isfinite(g["metres"]).all(1).sum()), feet=int(g["feet"].sum()), rolling=int(g["rolling"].sum()))
            kind = np.where(g["feet"], "feet", "rolling")
            pick = ST.pick_frames(p["frames"], g["ground"], kind, N_PER_HALF, MIN_GAP, seed=(int(m) if m.isdigit() else len(m)) + (h == "2nd"))
            if not pick: hs["status"] = "no ground balls"; continue
            try: vp = fetch(vids[h])
            except Exception as e: log(f"{m} {h}: video {e!r}"); hs["status"] = "video failed"; continue
            cap = cv2.VideoCapture(vp); W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or XML_FPS; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            hs.update(video_wh=[W, H], video_fps=round(fps, 3), video_frames=nfr)
            if cal is None:
                try: cal = ST.calibrate(pitch, image, W, H); ms["rms_px"] = round(cal["rms"], 2)
                except cv2.error as e: log(f"{m}: calibration refused"); ms["status"] = "calibration refused"; drop(vp); break
            bg, _ = ST.background(vp, 40); sv = list(np.linspace(nfr * 0.1, nfr * 0.9, 12).astype(int))
            masks = {k: ST.fg_mask(v, bg) for k, v in read_frames(vp, sv).items()}
            shift, hit, hit0 = ST.sync_shift(cal, p, m, masks); hs.update(sync_shift=int(shift), sync_hit=round(hit, 3), sync_hit_at_0=round(hit0, 3))
            if hit < MIN_HIT:
                log(f"{m} {h}: players do not line up at any shift (best {hit:.2f} at {shift}) -> skipped"); hs["status"] = "sync not found"; drop(vp); continue
            vf = {i: int(round((p["frames"][i] - off - shift) * fps / XML_FPS)) for i in pick}
            imgs = read_frames(vp, [v for v in vf.values() if 0 <= v < max(nfr, 1)]); hs["frames_read"] = len(imgs)
            bxy = ST.project(cal, g["metres"][pick]); bpx = ST.ball_px_size(cal, g["metres"][pick])
            nsnap = 0; rng = np.random.default_rng(len(labels))
            for j, i in enumerate(pick):
                img = imgs.get(vf[i])
                if img is None or not (0 <= bxy[j, 0] < W and 0 <= bxy[j, 1] < H): continue
                sx, sy, c, found = ST.snap(img, bxy[j], float(np.clip(2.5 * bpx[j], 10, 30)), bpx[j]); nsnap += found
                lab = dict(match=m, half=h, video=vids[h], frame=vf[i], xml_frame=int(p["frames"][i]), sync_shift=int(shift), x_raw=round(float(bxy[j, 0]), 1),
                           y_raw=round(float(bxy[j, 1]), 1), x=round(sx if found else float(bxy[j, 0]), 1), y=round(sy if found else float(bxy[j, 1]), 1),
                           snapped=found, snap_px=round(float(np.hypot(sx - bxy[j, 0], sy - bxy[j, 1])) if found else 0.0, 1), contrast=round(c, 1),
                           ball_px=round(float(bpx[j]), 1), kind=str(kind[i]), d_player_m=round(float(g["d_player"][i]), 2),
                           speed_ms=None if not np.isfinite(g["speed"][i]) else round(float(g["speed"][i]), 2),
                           pitch_m=[round(float(v), 2) for v in g["metres"][i]], W=W, H=H)
                labels.append(lab); X.append(ST.crop(img, (lab["x"], lab["y"]), 64)); tiles.append(tile(img, lab))
                for _ in range(20):                                           # a ball-free crop from the same frame
                    q = np.array([[rng.uniform(2, ST.PITCH_W - 2), rng.uniform(2, ST.PITCH_H - 2)]])
                    if np.linalg.norm(q - g["metres"][i]) < 6: continue
                    qx = ST.project(cal, q)[0]
                    if 32 <= qx[0] < W - 32 and 32 <= qx[1] < H - 32: N.append(ST.crop(img, qx, 64)); break
            hs.update(labels=sum(1 for l in labels if l["match"] == m and l["half"] == h), snapped=int(nsnap))
            if h == "1st" and imgs:                                           # overview: everybody projected on one frame
                i = pick[len(pick) // 2]; img = imgs.get(vf[i])
                if img is not None:
                    v = img.copy(); pl = ST.project(cal, ST.to_metres(p["players"][i], m, "player")) if len(p["players"][i]) else np.zeros((0, 2))
                    for q in pl: cv2.circle(v, (int(q[0]), int(q[1])), 14, (0, 255, 0), 3)
                    b = ST.project(cal, g["metres"][[i]])[0]; cv2.circle(v, (int(b[0]), int(b[1])), 18, (0, 255, 255), 3)
                    s = 1600 / v.shape[1]; cv2.imwrite(f"{OUT}/over_{m}.jpg", cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            drop(vp); log(f"{m} {h}: {hs.get('labels')} labels, snapped {nsnap}, rms {ms.get('rms_px')}, offset {off}")
        sheet(tiles, f"{OUT}/sheet_{m}.jpg"); ms.setdefault("status", "ok")
        json.dump(labels, open(f"{OUT}/labels.json", "w"), indent=0); json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    if X: np.savez_compressed(f"{OUT}/crops64.npz", X=np.array(X, np.uint8), N=np.array(N, np.uint8) if N else np.zeros((0, 64, 64, 3), np.uint8))
    summ.update(labels=len(labels), snapped=sum(l["snapped"] for l in labels), negatives=len(N), minutes=round((time.time() - t0) / 60, 1),
                kinds={k: sum(l["kind"] == k for l in labels) for k in ("feet", "rolling")})
    json.dump(labels, open(f"{OUT}/labels.json", "w"), indent=0); json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    log(f"done: {len(labels)} labels ({summ['snapped']} snapped), {len(N)} ball-free crops")

if __name__ == "__main__":
    main()
