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

def build_dataset(solution_json, frame_zips, out_dir, val_every=6, margin=4):
    sol = json.load(open(solution_json)); cam = sol["camera"]; W0, H0 = sol["image_size"]; kp = pitch_keypoints(*sol["pitch"])
    P = np.array([kp[n] for n in KEYPOINT_NAMES]); zips = {s: zipfile.ZipFile(p) for s, p in frame_zips.items()}; n_tr = n_va = 0; labels_per = []
    for i, fr in enumerate(sol["frames"]):
        split = "val" if i % val_every == val_every - 1 else "train"
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
