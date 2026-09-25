"""Full-match camera calibration from the line network (25 Sep).

Every processed frame (1 per second by default): the network paints the lines, the pose is refined from the previous
one (fast). Every `anchor_s` seconds, and whenever the refined pose looks doubtful, the frame is placed FRESH with the
cold search (no starting guess), polished on the painted pixels, and compared with the tracked pose. A frame is
CONFIDENT only if its own fit says so and it agrees with its neighbours; otherwise it is UNSURE and never guessed.

Self-grading: Daniel's clicked moments are inside the match. Their frames are processed exactly and graded against
his clicks (distance in px at 1280), so a full run comes with 56 checkpoints, not 9.

Output per chunk: a list of {t, pose, confident, cost, why} plus the checkpoint grades. `strip()` draws a contact sheet.
The network is injected (predict_fn) so the whole logic is testable offline with rendered masks."""
import json, time, numpy as np, cv2
from . import lines as LN

def hard_moment(prev_pose, pose, dt):
    """what makes a moment hard: a fast pan or a deep zoom (reported separately, so failures can be placed)"""
    tags = []
    if pose[3] > 2200: tags.append("zoom")
    if prev_pose is not None and dt > 0 and abs(np.degrees(pose[0] - prev_pose[0])) / dt > 3.0: tags.append("fast pan")
    return tags

def place(mask, camera, prev=None, big=None, snap=True, jump_px=40.0, agree_px=8.0):
    """one frame: refine from prev if given, cold-place if not (or if the refine is doubtful), polish, decide confidence.
    Returns (pose or None, info). big = full-size frame for the painted-pixel polish."""
    why = []; pose = None; info = {}
    if prev is not None:
        pose, info = LN.fit_pose(mask, camera, init=prev)
        if pose is None: why.append("no lines")
        elif not info.get("confident"): why.append("refine doubtful")
    if pose is None or why:                                                # fresh placement, no starting guess
        cold, cinfo = LN.fit_pose(mask, camera)
        if cold is None: return None, {"confident": False, "why": why + ["no lines"], "cost": None}
        if pose is not None and cinfo.get("confident") and LN.pose_error(camera, cold, pose)["median_px"] > jump_px:
            why.append("cold and tracked disagree")
        if cinfo.get("confident") or pose is None: pose, info = cold, cinfo
    move = None; before = None
    if snap and big is not None and pose is not None:                    # second opinion: the exact painted pixels
        before = np.asarray(pose, float); pose = np.array(LN.snap(big, camera, pose)); move = LN.pose_error(camera, pose, before)["median_px"]
        if move > agree_px: why.append("lines and paint disagree")        # two independent estimates apart: not to be trusted
    conf = bool(info.get("confident")) and not any(w in ("cold and tracked disagree", "no lines", "lines and paint disagree") for w in why)
    return pose, {"confident": conf, "why": why, "cost": info.get("cost"), "classes": info.get("classes_seen"), "unexplained": info.get("unexplained"),
                  "snap_move_px": None if move is None else round(float(move), 1), "before_snap": None if before is None else [float(v) for v in before]}

def run_chunk(frames, camera, predict_fn, t0=0.0, fps=1.0, anchor_s=5.0, checkpoints=None, snap=True, log=None, max_jump_px=60.0, agree_px=8.0, two_way=True):
    """frames: iterable of (t, image) at ~fps. Forward pass: anchors every anchor_s (fresh placement + polish), tracked
    frames in between. Backward pass (two_way): each in-between frame is re-tracked from the NEXT anchor; forward and
    backward must agree within agree_px (at 1280) for the frame to be confident, and the average is used. checkpoints:
    {t: clicks} graded on the exact frame. Returns (rows, grades)."""
    rows, grades = [], []; prev = None; prev_t = None; last_anchor = -1e9; masks = []
    for t, img in frames:
        small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA) if img.shape[1] != 640 else img
        mask = predict_fn(small); big = cv2.resize(img, (1280, 720)) if img.shape[1] != 1280 else img
        anchor = (t - last_anchor) >= anchor_s
        is_cp = bool(checkpoints) and any(abs(tc - t) <= 1e-3 for tc in checkpoints)
        pose, info = place(mask, camera, prev=None if anchor else prev, big=big if snap else None, keep_unsnapped=is_cp)
        if anchor and pose is not None:
            last_anchor = t
            if prev is not None and LN.pose_error(camera, pose, prev)["median_px"] > max_jump_px and info["confident"]:
                info["why"].append("jump from track"); info["confident"] = False
        tags = hard_moment(prev, pose, (t - prev_t) if prev_t is not None else 0) if pose is not None else []
        rows.append({"t": round(float(t), 3), "pose": None if pose is None else [float(v) for v in pose], "confident": bool(info["confident"]),
                     "anchor": bool(anchor), "why": list(info["why"]), "cost": info["cost"], "hard": tags}); masks.append(mask)
        if pose is not None: prev, prev_t = pose, t
        if log and len(rows) % 60 == 0: log(f"  t={t:.0f}s: {sum(r['confident'] for r in rows)}/{len(rows)} confident")
    if two_way: backward_pass(rows, masks, camera, agree_px)
    if checkpoints:
        for r in rows:
            for tc, clicks in checkpoints.items():
                if abs(tc - r["t"]) <= 1e-3 and r["pose"] is not None and clicks:
                    e = LN.click_error(camera, r["pose"], clicks)
                    if np.isfinite(e["median_px"]): grades.append({"t": tc, "frame_t": r["t"], "median_px": round(e["median_px"], 1), "confident": bool(r["confident"])})
    return rows, grades

def backward_pass(rows, masks, camera, agree_px=8.0):
    """re-track every non-anchor frame from the NEXT anchor backwards; a frame is confident only if the two directions
    agree within agree_px (px at 1280); the pose becomes their average (25 Sep: 4 silent 17-41 px misses in the full
    match were all tracked frames whose fit looked fine on its own)."""
    n = len(rows); nxt = None
    for i in range(n - 1, -1, -1):
        r = rows[i]
        if r["anchor"] or r["pose"] is None:
            nxt = np.array(r["pose"]) if (r["pose"] is not None and r["anchor"]) else nxt; continue
        if nxt is None: r["why"].append("no anchor after"); r["confident"] = False; continue
        back, binfo = LN.fit_pose(masks[i], camera, init=nxt)
        if back is None: r["why"].append("backward lost"); r["confident"] = False; continue
        fwd = np.array(r["pose"]); d = LN.pose_error(camera, back, fwd)["median_px"]; r["fwd_bwd_px"] = round(float(d), 1)
        if d > agree_px: r["why"].append("forward/backward disagree"); r["confident"] = False
        else:
            avg = (fwd + back) / 2; avg[3] = np.sqrt(fwd[3] * back[3]); r["pose"] = [float(v) for v in avg]
        nxt = back

def frame_times(duration_s, fps=1.0, checkpoint_ts=()):
    """the moments to process: a regular grid plus the exact checkpoint moments"""
    ts = sorted(set(np.round(np.arange(0, duration_s, 1.0 / fps), 3).tolist()) | {round(float(t), 3) for t in checkpoint_ts})
    return ts

def read_frames(video, times, log=None):
    """yield (t, frame) for the given times from a video, reading sequentially (no random seeks)"""
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97; want = [int(round(t * fps)) for t in times]; i = 0; k = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0]); k = want[0]
    while i < len(want):
        ok, f = cap.read()
        if not ok: break
        while i < len(want) and want[i] == k: yield times[i], f; i += 1
        k += 1
    cap.release()

def summarize(rows, grades, near_px=10.0):
    n = len(rows); c = sum(r["confident"] for r in rows); hard = sum(1 for r in rows if r["hard"]); hard_c = sum(1 for r in rows if r["hard"] and r["confident"])
    unsure_runs = []; run = None
    for r in rows:
        if not r["confident"]:
            if run and r["t"] - run[1] <= 2.0: run[1] = r["t"]
            else: run = [r["t"], r["t"]]; unsure_runs.append(run)
    long_gaps = [(a, b) for a, b in unsure_runs if b - a >= 10]
    g = [x["median_px"] for x in grades]
    return {"frames": n, "confident": c, "confident_share": round(c / max(n, 1), 3), "hard_moments": hard, "hard_confident": hard_c,
            "unsure_stretches_over_10s": len(long_gaps), "longest_unsure_s": max([b - a for a, b in unsure_runs], default=0),
            "checkpoints": len(grades), "checkpoints_within_10px": sum(1 for x in g if x <= near_px), "checkpoint_median_px": float(np.median(g)) if g else None,
            "checkpoints_confident_and_wrong": sum(1 for x in grades if x["confident"] and x["median_px"] > near_px)}

def strip(video_or_frames, rows, camera, n=20, out_path=None):
    """n pictures spread over the run with the fitted lines drawn and the verdict written on them (for Daniel's eyes)"""
    pick = [rows[i] for i in np.linspace(0, len(rows) - 1, n).astype(int)] if rows else []
    frames = dict(video_or_frames) if not isinstance(video_or_frames, str) else None; tiles = []
    if frames is None:
        cap = cv2.VideoCapture(video_or_frames); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    for r in pick:
        if frames is not None: img = frames.get(r["t"])
        else: cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r["t"] * fps))); ok, img = cap.read(); img = img if ok else None
        if img is None: continue
        img = cv2.resize(img, (640, 360)); o = LN.draw_pose(img, camera, r["pose"]) if r["pose"] else img.copy()
        txt = f"t={r['t']:.0f}s " + ("OK" if r["confident"] else "UNSURE: " + ", ".join(r["why"] or ["?"])) + (" [" + ", ".join(r["hard"]) + "]" if r["hard"] else "")
        cv2.putText(o, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(o, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); tiles.append(o)
    if frames is None: cap.release()
    if not tiles: return None
    rows_ = [np.hstack(tiles[i:i + 2] + [np.zeros((360, 640, 3), np.uint8)] * (2 - len(tiles[i:i + 2]))) for i in range(0, len(tiles), 2)]
    sheet = np.vstack(rows_)
    if out_path: cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return sheet


def calibrate_video(video, weights, camera, t0, t1, fps=1.0, anchor_s=5.0, checkpoints=None, log=print, snap=True, encoder="resnet34", pictures=None):
    """one stretch [t0, t1) of a match with the real network (CPU is fine). Returns (rows, grades)."""
    from . import linetrain as LT
    model, device = LT.load(weights, encoder, device="cpu")
    predict = lambda img: LT.predict(model, img, device)
    cps = {t: c for t, c in (checkpoints or {}).items() if t0 <= t < t1}
    times = [t for t in frame_times(t1, fps, cps) if t >= t0]
    if log: log(f"  stretch {t0:.0f}-{t1:.0f}s: {len(times)} frames, {len(cps)} checkpoints")
    return run_chunk(read_frames(video, times), camera, predict, fps=fps, anchor_s=anchor_s, checkpoints=cps, snap=snap, log=log, pictures=pictures)

def checkpoints_from_clicks(root):
    """{t: clicks} for every clicked frame (both sessions)"""
    import os
    out = {}
    for s in ("s1", "s2"):
        p = f"{root}/results/labels/SFKBP1109_points_{s}.json"
        if os.path.exists(p):
            for name, v in json.load(open(p)).items(): out[round(float(name[3:-4]), 3)] = v["pairs"]
    return out
