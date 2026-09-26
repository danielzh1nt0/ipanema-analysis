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

def calibration_for_clip(rows_path, n_frames, fps, w, h, offset_s=0.0, L=106.0, W=64.0, bridge_s=4.0, draw_bridge_s=2.0, log=print):
    """H per frame. Confident seconds (Daniel's braver setting) are trusted; a gap of unsure seconds up to bridge_s long is
    bridged by interpolating between the confident poses on either side (the camera moves smoothly; a 1-3 s bridge is far
    better than borrowing a neighbour's camera, which put the ball off the pitch and invented dead balls / restarts).
    Lines are drawn for bridges up to draw_bridge_s; longer bridges and unbridgeable stretches (nearest confident camera
    borrowed, metres unreliable) are listed in cal["unsure"] and get no drawn lines."""
    d = json.load(open(rows_path)); cam = d["camera"]; rows = sorted(d["rows"], key=lambda r: r["t"])
    ts = np.array([r["t"] for r in rows]); ok = np.array([brave(r) for r in rows]); poses = np.array([r["pose"] for r in rows], float)
    conf_idx = np.nonzero(ok)[0]
    if not len(conf_idx): raise RuntimeError("line calibration: no confident second in this match")
    H = {}; unsure = set(); bridged = 0
    for k in range(n_frames):
        t = offset_s + k / fps
        a = conf_idx[np.searchsorted(ts[conf_idx], t, side="right") - 1] if t >= ts[conf_idx[0]] else None      # last confident at or before t
        nb = np.searchsorted(ts[conf_idx], t, side="left"); b = conf_idx[nb] if nb < len(conf_idx) else None    # first confident at or after t
        if a is None or b is None: pa, pb, ta, tb = (poses[b], poses[b], ts[b], ts[b]) if a is None else (poses[a], poses[a], ts[a], ts[a]); gap = float("inf") if (a is None or b is None) and abs(t - (ts[a] if a is not None else ts[b])) > 1.5 else 0.0
        else: pa, pb, ta, tb = poses[a], poses[b], ts[a], ts[b]; gap = tb - ta
        if gap > bridge_s: unsure.add(k)                                       # borrow the nearer confident pose, flagged
        u = 0.0 if tb <= ta else float(np.clip((t - ta) / (tb - ta), 0, 1)); pose = (1 - u) * pa + u * pb
        if gap > draw_bridge_s and gap > 1.0 + 1e-6: unsure.add(k); bridged += (gap <= bridge_s)
        H[k] = homography(cam, pose, w, h, L, W)
    valid = sorted(k for k in H if H[k] is not None)
    for k in [k for k in H if H[k] is None]: H[k] = H[min(valid, key=lambda v: abs(v - k))]; unsure.add(k)
    cov = 1 - len(unsure) / max(n_frames, 1)
    log(f"calibration: from the line model, {cov:.0%} of frames trusted (lines drawn); {len(unsure)} frames bridged or borrowed, no drawn lines")
    return {"coverage": cov, "frozen": len(unsure), "H": H, "L": L, "W": W, "unsure": unsure, "camera": cam, "source": "lines"}
