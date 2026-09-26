"""The line-based full-match calibration (matchcal rows: one camera pose per second) feeding the app pipeline (26 Sep).

run.prepare needs H[k]: a 3x3 pitch-metres -> pixels homography for every frame k of the clip. A pinhole camera looking
at the flat pitch IS a homography, so each pose gives an exact H (four projected points -> getPerspectiveTransform).
Between two consecutive confident seconds the pose is interpolated; a frame with an unsure neighbour gets the nearest
confident H (so tracking still has metres) but is listed in cal["unsure"], and export writes no pitch lines for it.

The rows file: calibration/<match>_lines_match.json = {"camera": {...}, "rows": [...]} (from results/match/<n>/rows.json).
Clips cut from the match are named <match>_s<start seconds>; the offset is read from the name."""
import os, re, json, numpy as np, cv2
from . import lines as LN

FB_PX = 15.0

def brave(q):
    """the setting Daniel chose (25 Sep): forward/backward agree within 15 px; 'lines and paint disagree' is not a veto"""
    return q.get("pose") is not None and (q.get("fwd_bwd_px") is None or q["fwd_bwd_px"] <= FB_PX) \
        and all(w.startswith("lines and paint") or w.startswith("forward/backward") for w in q.get("why", []))

def homography(camera, pose, w, h, L=106.0, W=64.0):
    """exact, in double precision: pixel = K R (P - C) with P on z = 0  ->  H = K [r1 r2 | -R C]"""
    R = LN._rot(*pose[:3]) @ LN._base(*camera["base_tilt"]); C = np.asarray(camera["C"], float); f = pose[3] * w / 1280.0
    M = np.column_stack([R[:, 0], R[:, 1], -R @ C]); K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float); H = K @ M
    if abs(np.linalg.det(H)) < 1e-12: return None
    return H / H[2, 2] if abs(H[2, 2]) > 1e-9 else H

def find_rows(root, match_id):
    """(base match id, offset seconds, rows path) or None"""
    m = re.match(r"^(.*?)_s(\d+)$", match_id); base, off = (m.group(1), float(m.group(2))) if m else (match_id, 0.0)
    for d in (root, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration")):
        p = os.path.join(d, "calibration", f"{base}_lines_match.json") if d == root else os.path.join(d, f"{base}_lines_match.json")
        if os.path.exists(p): return base, off, p
    return None

def calibration_for_clip(rows_path, n_frames, fps, w, h, offset_s=0.0, L=106.0, W=64.0, log=print):
    d = json.load(open(rows_path)); cam = d["camera"]; rows = sorted(d["rows"], key=lambda r: r["t"])
    ts = np.array([r["t"] for r in rows]); ok = np.array([brave(r) for r in rows]); poses = [r["pose"] for r in rows]
    H = {}; unsure = set(); cache = {}
    for k in range(n_frames):
        t = offset_s + k / fps; j = int(np.searchsorted(ts, t, side="right")) - 1
        a, b = (j, j + 1) if 0 <= j < len(rows) - 1 else (min(max(j, 0), len(rows) - 1), min(max(j, 0), len(rows) - 1))
        if ok[a] and ok[b] and ts[b] - ts[a] <= 2.0 + 1e-6:
            u = 0.0 if b == a else float(np.clip((t - ts[a]) / (ts[b] - ts[a]), 0, 1)); pose = [(1 - u) * pa + u * pb for pa, pb in zip(poses[a], poses[b])]
            H[k] = homography(cam, pose, w, h, L, W)
        else: H[k] = None
        if H[k] is None: unsure.add(k)
    valid = sorted(k for k in H if H[k] is not None)
    if not valid: raise RuntimeError("line calibration: no confident second in this clip")
    for k in unsure: H[k] = H[min(valid, key=lambda v: abs(v - k))]
    cov = 1 - len(unsure) / max(n_frames, 1)
    log(f"calibration: from the line model, {cov:.0%} of frames confident ({len(unsure)} frames borrow the nearest confident camera and get no drawn lines)")
    return {"coverage": cov, "frozen": len(unsure), "H": H, "L": L, "W": W, "unsure": unsure, "camera": cam, "source": "lines"}
