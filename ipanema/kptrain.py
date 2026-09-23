"""Pitch-point detector training data from Daniel's clicks (YOLO pose format).

Each clicked frame has a solved camera pose (calibration/panorama/<match>_clicks_solution.json), so EVERY standard pitch
point in view can be labelled, not only the clicked ones. One "pitch" object per image; keypoints in the fixed order of
KEYPOINT_NAMES; visibility 2 = in view, 0 = not in view. Every `val_every`-th frame is held back for grading."""
import os, json, zipfile, numpy as np, cv2
from .label import pitch_keypoints

KEYPOINT_NAMES = sorted(pitch_keypoints().keys())

def _rot(pan, tilt, roll):
    d = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])
    r = np.cross([0, 0, 1.0], d); r /= np.linalg.norm(r); R = np.vstack([r, np.cross(d, r), d])
    c, s = np.cos(roll), np.sin(roll); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R

def _base(bx, by):
    cx, sx, cy, sy = np.cos(bx), np.sin(bx), np.cos(by), np.sin(by)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

def project(camera, pose, P, w, h):
    """pitch metres -> pixels (NaN behind the camera), the same model the pipeline uses"""
    R = _rot(*pose[:3]) @ _base(*camera["base_tilt"]); f = pose[3]
    c = (np.column_stack([np.asarray(P, float), np.zeros(len(P))]) - np.asarray(camera["C"], float)) @ R.T
    out = np.column_stack([w / 2 + f * c[:, 0] / c[:, 2], h / 2 + f * c[:, 1] / c[:, 2]]); out[c[:, 2] <= 1e-6] = np.nan
    return out

def merge_aligned(solution_json, aligned):
    """add lined-up frames ({session: align json}) to the click solution; poses are stored per 1280x720 frame"""
    sol = json.load(open(solution_json)); have = {(f["session"], f["frame"]) for f in sol["frames"]}
    for s, path in aligned.items():
        if not os.path.exists(path): continue
        for name, v in json.load(open(path)).items():
            if v.get("pose") and not v.get("skipped") and (s, name) not in have:
                w = (v.get("size") or [1280, 720])[0]; p = list(v["pose"]); p[3] *= 1280.0 / w
                sol["frames"].append({"session": s, "frame": name, "pose": p, "source": "aligned"})
    return sol

def build_dataset(solution_json, frame_zips, out_dir, val_every=6, margin=4):
    sol = json.load(open(solution_json)) if isinstance(solution_json, str) else solution_json
    cam = sol["camera"]; W0, H0 = sol["image_size"]; kp = pitch_keypoints(*sol["pitch"])
    P = np.array([kp[n] for n in KEYPOINT_NAMES]); zips = {s: zipfile.ZipFile(p) for s, p in frame_zips.items()}; n_tr = n_va = 0; labels_per = []
    for i, fr in enumerate(sol["frames"]):
        split = "val" if (i % val_every == val_every - 1 and not fr.get("source")) else "train"   # only clicked frames are ever held back
        img = cv2.imdecode(np.frombuffer(zips[fr["session"]].read(fr["frame"]), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]; q = project(cam, fr["pose"], P, W0, H0) * np.array([w / W0, h / H0])
        vis = np.isfinite(q).all(1) & (q[:, 0] >= margin) & (q[:, 0] < w - margin) & (q[:, 1] >= margin) & (q[:, 1] < h - margin)
        if vis.sum() < 2: continue
        name = f"{fr['session']}_{fr['frame'][:-4]}"
        for sub in ("images", "labels"): os.makedirs(f"{out_dir}/{sub}/{split}", exist_ok=True)
        cv2.imwrite(f"{out_dir}/images/{split}/{name}.jpg", img)
        kps = " ".join(f"{(q[j, 0] / w if vis[j] else 0):.6f} {(q[j, 1] / h if vis[j] else 0):.6f} {2 if vis[j] else 0}" for j in range(len(P)))
        open(f"{out_dir}/labels/{split}/{name}.txt", "w").write(f"0 0.5 0.5 1.0 1.0 {kps}\n")
        labels_per.append(int(vis.sum())); n_tr += split == "train"; n_va += split == "val"
    open(f"{out_dir}/data.yaml", "w").write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nkpt_shape: [{len(P)}, 3]\nflip_idx: {list(range(len(P)))}\nnames:\n  0: pitch\n")
    print(f"dataset: {n_tr} train + {n_va} held-back frames, {np.mean(labels_per):.1f} labelled points per frame (you clicked ~6), {len(P)} point types -> {out_dir}")
    return f"{out_dir}/data.yaml"


def evaluate(weights, ds_dir, conf=0.5, near_px=15.0):
    """Grade a trained model on the HELD-BACK frames only: for each labelled point, is the model's point within near_px?
    Also: on how many frames does it find at least 2 correct points (with the camera known, 2 points place the frame)."""
    from ultralytics import YOLO
    m = YOLO(weights); errs = []; placed = 0; frames = 0; wrong = 0
    for f in sorted(os.listdir(f"{ds_dir}/images/val")):
        img = cv2.imread(f"{ds_dir}/images/val/{f}"); h, w = img.shape[:2]; frames += 1
        lab = np.array(open(f"{ds_dir}/labels/val/{f[:-4]}.txt").read().split()[5:], float).reshape(-1, 3)
        r = m(img, imgsz=max(h, w), conf=0.1, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0: continue
        xy = r.keypoints.xy[0].cpu().numpy(); kc = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        ok_here = 0
        for j in range(len(lab)):
            if kc[j] < conf: continue
            if lab[j, 2] == 0: wrong += 1; continue                          # confident about a point that isn't in view
            e = float(np.hypot(xy[j, 0] - lab[j, 0] * w, xy[j, 1] - lab[j, 1] * h)); errs.append(e); ok_here += e <= near_px
        placed += ok_here >= 2
    e = np.array(errs) if errs else np.array([np.inf])
    return {"held_back_frames": frames, "frames_with_2+_correct_points": placed, "points_median_px": round(float(np.median(e)), 1),
            "points_within_px": f"{100 * (e <= near_px).mean():.0f}% within {near_px:.0f} px", "confident_but_not_in_view": wrong}


def propagate(video, solution_json, base_json, out_zip, out_json, secs=3.0, fps_out=5.0, width=1280, val_every=6, log=print):
    """Carry each TRAINING clicked frame's calibration +-secs with the tracker (held-back frames are never used), saving the
    frames and their proposed poses for a yes/no review. Works on the full match video (e.g. from Drive in Colab)."""
    import zipfile
    from . import ptz, fccam as FC
    sol = json.load(open(solution_json)); base = json.load(open(base_json)); C = base["C"]; ptz.set_base_tilt(*base["base_tilt"]); FC.set_base_tilt(*base["base_tilt"])
    cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    z = zipfile.ZipFile(out_zip, "w"); poses = {}; n_seeds = 0
    for i, fr in enumerate(sol["frames"]):
        if i % val_every == val_every - 1 or fr.get("source") == "aligned": continue          # held-back frames stay untouched
        t0 = float(fr["frame"][3:-4]); ts = np.arange(t0 - secs, t0 + secs + 1e-6, 1.0 / fps_out); ts = ts[ts >= 0]
        frames = []
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * vfps))); ok, f = cap.read()
            frames.append(cv2.resize(f, (width, int(round(f.shape[0] * width / f.shape[1])))) if ok else None)
        if any(f is None for f in frames): continue
        h, w = frames[0].shape[:2]; k0 = int(np.argmin(np.abs(ts - t0))); p0 = np.array(fr["pose"], float); p0[3] *= w / 1280.0
        rl = lambda f_, p_: FC.refine(f_, C, 106.0, 64.0, p_)
        fwd, _ = ptz.track_sequence(frames[k0:], C, p0, w, h, refine_lines=rl); bwd, _ = ptz.track_sequence(frames[k0::-1], C, p0, w, h, refine_lines=rl)
        seq = list(reversed(bwd[1:])) + list(fwd); n_seeds += 1
        for t, f, p in zip(ts, frames, seq):
            name = f"fc_{t:08.3f}.jpg"
            if name in poses or abs(t - t0) < 1e-6: continue                                    # the clicked frame itself is already in the set
            z.writestr(name, cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
            poses[name] = {"pose": [float(v) for v in p], "size": [w, h], "seed": fr["frame"]}
        log(f"  seed {n_seeds}: {fr['frame']} -> {len(poses)} proposed frames so far")
        json.dump(poses, open(out_json, "w"))                                                   # saved as it goes
    z.close(); cap.release(); log(f"propagated from {n_seeds} clicked frames -> {len(poses)} proposed frames"); return len(poses)

def merge_reviewed(sol, proposals_json, review_json, session):
    """add proposed frames the reviewer said YES to"""
    prop = json.load(open(proposals_json)); rev = json.load(open(review_json)) if os.path.exists(review_json) else {}
    for name, v in prop.items():
        if rev.get(name) == "yes":
            p = list(v["pose"]); p[3] *= 1280.0 / v["size"][0]; sol["frames"].append({"session": session, "frame": name, "pose": p, "source": "reviewed"})
    return sol


def propose_random(video, base_json, out_zip, out_json, n_candidates=900, play=((0.0, 51 * 60.0), (62 * 60 + 13.0, 99 * 60.0)),
                   avoid=(), avoid_s=5.0, width=1280, fit_width=960, seed=0, log=print):
    """VARIETY: calibrate moments spread over the whole match with the single-frame line fit (known camera base), keep
    only those the judge accepts, and save them for the yes/no review (same format as propagate). `avoid` = times (s)
    already labelled or held back; nothing within avoid_s of them is proposed."""
    import zipfile
    from . import fccam as FC
    from .calcheck import judge_frame
    base = json.load(open(base_json)); C = base["C"]; FC.set_base_tilt(*base["base_tilt"])
    rng = np.random.RandomState(seed); total = sum(b - a for a, b in play); times = []
    for u in np.sort(rng.uniform(0, total, n_candidates)):
        for a, b in play:
            if u < b - a: times.append(a + u); break
            u -= b - a
    avoid = np.array(sorted(avoid), float)
    times = [t for t in times if not len(avoid) or np.min(np.abs(avoid - t)) > avoid_s]
    cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    z = zipfile.ZipFile(out_zip, "w"); out = {}; tried = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * vfps))); ok, f = cap.read()
        if not ok: continue
        tried += 1; big = cv2.resize(f, (width, int(round(f.shape[0] * width / f.shape[1])))); small = cv2.resize(f, (fit_width, int(round(f.shape[0] * fit_width / f.shape[1]))))
        H, info = FC.fit(small, C, 106.0, 64.0, tilt_range=(0.5, 45), f_range=(500, 4500), coarse=(3.0, 1.5, 10))
        if judge_frame(small, H, 106.0, 64.0)["verdict"] != "good": continue
        pose = [float(np.radians(info["pan_deg"])), float(np.radians(info["tilt_deg"])), float(np.radians(info["roll_deg"])), float(info["zoom"]) * width / fit_width]
        name = f"fc_{t:08.3f}.jpg"; z.writestr(name, cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
        out[name] = {"pose": pose, "size": [big.shape[1], big.shape[0]], "seed": "auto"}
        if len(out) % 20 == 0: json.dump(out, open(out_json, "w")); log(f"  {len(out)} proposed from {tried} moments tried")
    z.close(); cap.release(); json.dump(out, open(out_json, "w"))
    log(f"proposed {len(out)} new moments out of {tried} tried ({100 * len(out) / max(1, tried):.0f}%), spread over the match")
    return len(out)

def labelled_times(solution_json):
    """times (s) of every clicked frame, to keep new proposals away from them"""
    return [float(f["frame"][3:-4]) for f in json.load(open(solution_json))["frames"]]
