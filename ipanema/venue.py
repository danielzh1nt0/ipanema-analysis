"""New venue: solve the camera base from the line network alone (26 Sep, first use: Norrvikens IP, SFK-AIK).

The camera is a Veo on a mast by the halfway line, so the base is searched near there: x ~ L/2, y a few metres behind
the near touchline, height 4-6 m. Frames spread over the match are placed with fit_pose at each candidate base; the base
whose placements explain the painted-line masks best (lowest summed two-way cost over confident frames) wins, then a
continuous refine of base + all poses on the sharp painted pixels. Held-out check: solve on half the frames, cost on the
other half must improve over the Edsberg base carried over as-is.

No clicks needed. Daniel checks a strip. Everything is graded by masks the network painted, not by anyone's opinion."""
import json, numpy as np, cv2
from scipy.optimize import least_squares, minimize
from . import lines as LN

def candidate_bases(L=106.0, W=64.0, dx=(-6, 6, 3.0), dy=(1.5, 7.5, 1.5), dz=(-6.5, -3.5, 1.0)):
    """camera positions to try: along the halfway line, behind the near touchline, 3.5-6.5 m up (z is DOWN-positive)"""
    out = []
    for x in np.arange(L / 2 + dx[0], L / 2 + dx[1] + 1e-9, dx[2]):
        for y in np.arange(W + dy[0], W + dy[1] + 1e-9, dy[2]):
            for z in np.arange(dz[0], dz[1] + 1e-9, dz[2]): out.append([float(x), float(y), float(z)])
    return out

def _cost_of_base(masks, C, base_tilt=(0.0, 0.0), keep=4, init=None):
    """mean two-way cost of the best poses under base C. With init (poses from another base) only a local refine is
    run per frame (~0.3 s) instead of the full cold search (~6 s)."""
    cam = {"C": list(C), "base_tilt": list(base_tilt)}; tot = 0.0; n = 0; poses = []
    for k, m in enumerate(masks):
        p0 = init[k] if init is not None else None
        if init is not None and p0 is None: poses.append(None); continue
        p, info = LN.fit_pose(m, cam, keep=keep, init=p0)
        if p is None: poses.append(None); continue
        poses.append(p); tot += info["cost"]; n += 1
    return (tot / n if n else np.inf), poses, n

def solve_base(masks, ref_camera, log=print, coarse_frames=12, L=106.0, W=64.0):
    """masks: predicted class masks (640x360) from frames spread over the match. ref_camera: a known venue's base
    (Edsberg), used only to get first poses. Returns (camera, poses, report)."""
    sub = masks[:: max(1, len(masks) // coarse_frames)][:coarse_frames]
    _, p_ref, n_ref = _cost_of_base(sub, ref_camera["C"], tuple(ref_camera["base_tilt"]), keep=6)    # one cold fit per frame
    if log: log(f"  first poses from the reference base: {n_ref}/{len(sub)} frames placed")
    cands = candidate_bases(L, W); best = (np.inf, None, None); scores = []
    for i, C in enumerate(cands):
        c, pp, n = _cost_of_base(sub, C, init=p_ref); scores.append((c, C))
        if c < best[0]: best = (c, C, pp)
        if log and i % 25 == 24: log(f"  base search {i + 1}/{len(cands)}: best so far {np.round(best[1], 1)} cost {best[0]:.3f}")
    scores.sort(key=lambda s: s[0]); C0 = best[1]
    if log: log(f"  coarse best base {np.round(C0, 2)} (cost {best[0]:.3f}); runner-up {np.round(scores[1][1], 2)} (cost {scores[1][0]:.3f})")
    # all frames: cold fit under C0, then continuous refine of the base (x, y, z, tilt) with local per-frame refits
    _, poses, _ = _cost_of_base(masks, C0, keep=6); idx = [i for i, p in enumerate(poses) if p is not None]; cur = {i: poses[i] for i in idx}
    def total(x):
        camx = {"C": [float(v) for v in x[:3]], "base_tilt": [float(v) for v in x[3:5]]}; s = 0.0
        for i in idx:
            p, info = LN.fit_pose(masks[i], camx, init=cur[i]); cur[i] = p if p is not None else cur[i]
            s += info["cost"] if p is not None else 10.0
        return s / len(idx)
    x0 = np.r_[C0, 0.0, 0.0]
    r = minimize(total, x0, method="Nelder-Mead", options={"maxfev": 50, "xatol": 0.05, "fatol": 1e-3,
                 "initial_simplex": np.array([x0] + [x0 + d for d in (np.r_[1.5, 0, 0, 0, 0], np.r_[0, 1.0, 0, 0, 0], np.r_[0, 0, 0.6, 0, 0], np.r_[0, 0, 0, 0.01, 0], np.r_[0, 0, 0, 0, 0.01])])})
    x = r.x; camera = {"C": [float(v) for v in x[:3]], "base_tilt": [float(v) for v in x[3:5]]}; total(x)
    out_poses = [None] * len(masks)
    for i in idx: out_poses[i] = [float(v) for v in cur[i]]
    rep = {"coarse_best": [round(float(v), 2) for v in C0], "coarse_cost": round(float(best[0]), 3), "refined_cost": round(float(r.fun), 3), "refine_evals": int(r.nfev),
           "base": camera, "frames_used": len(idx), "of": len(masks), "runner_up": [round(float(v), 2) for v in scores[1][1]], "runner_up_cost": round(float(scores[1][0]), 3)}
    if log: log(f"  refined base {np.round(camera['C'], 2)} tilt {np.round(np.degrees(camera['base_tilt']), 2)} deg, cost {best[0]:.3f} -> {r.fun:.3f} on {len(idx)}/{len(masks)} frames")
    return camera, out_poses, rep

def heldout_check(masks, camera_new, camera_ref, log=print):
    """A/B halves: how well each base explains frames it did not see (cost per frame, lower is better)"""
    B = masks[1::2]; res = {}
    for name, cam in (("new", camera_new), ("reference", camera_ref)):
        c, _, n = _cost_of_base(B, cam["C"], tuple(cam["base_tilt"]), keep=4); res[name] = {"cost": round(float(c), 3), "frames": n}
    if log: log(f"  held-out half: new base cost {res['new']['cost']} vs reference (Edsberg) {res['reference']['cost']}")
    return res

def strip(frames, masks, camera, poses, out_path=None, n=20):
    pick = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int); tiles = []
    for i in pick:
        img = cv2.resize(frames[i], (640, 360)); o = LN.overlay(img, masks[i], 0.35)
        if poses[i] is not None: o = LN.draw_pose(o, camera, poses[i], thick=2)
        cv2.putText(o, f"#{i}" + ("" if poses[i] is not None else " no pose"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(o, f"#{i}" + ("" if poses[i] is not None else " no pose"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(o)
    rows = [np.hstack(tiles[k:k + 2] + [np.zeros((360, 640, 3), np.uint8)] * (2 - len(tiles[k:k + 2]))) for k in range(0, len(tiles), 2)]
    sheet = np.vstack(rows)
    if out_path: cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return sheet
