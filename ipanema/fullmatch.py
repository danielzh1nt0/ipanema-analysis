"""Full matches: cut into pieces, process the pieces in parallel (GPU), join them into one match and analyse it (CPU).

Frame alignment: a piece starting at start_s covers global frames round(start_s * fps) + k. Veo records at 29.97 fps,
so fps is read from the full video, never assumed. Tracker ids are made unique per piece; the cleanup step that
joins broken tracks (TR.clean, within 2.5 s) then runs on the joined match, so players carry across the cuts."""
import os, json, pickle, subprocess, numpy as np

PIECE_S = 300
ID_STRIDE = 100_000     # tracker ids of piece i become i * ID_STRIDE + id
PIECE_FILE = "piece_v2.pkl"   # v1 pieces were made with the fallback calibration (panorama registration crashed) and are not used

def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return n, fps

def plan(n_frames, fps, piece_s=PIECE_S):
    total = n_frames / fps; out = []; i = 0; s = 0
    while s < total - 1:
        out.append({"i": i, "start_s": s, "dur_s": min(piece_s, total - s), "offset": int(round(s * fps))}); i += 1; s += piece_s
    return out

def piece_id(match_id, i): return f"{match_id}_c{i:03d}"

def cut(full, dst, start_s, dur_s):
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000: return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", full, "-t", f"{dur_s:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", dst], check=True, capture_output=True)
    return dst

def process_piece(full, match_id, piece, S, log=print):
    """GPU part for one piece -> saves and returns the path of its pickle"""
    from .run import prepare
    pid = piece_id(match_id, piece["i"]); out = os.path.join(S.root, "cache", pid, PIECE_FILE)
    if os.path.exists(out): log(f"{pid}: cached"); return out
    src = cut(full, os.path.join(S.root, "videos", f"{pid}.mp4"), piece["start_s"], piece["dur_s"])
    ctx = prepare(src, pid, S, log=log, train_ball=False, debug=(piece["i"] in (0, 4, 10, 16)))
    keep = {"per": ctx["per"], "H": {k: (v if hasattr(v, "to_m") else np.asarray(v, np.float32)) for k, v in ctx["H"].items()}, "cands": ctx["cands"], "fps": ctx["fps"],
            "n": ctx["vi"]["n"], "L": ctx["L"], "W": ctx["W"], "coverage": ctx["cal"]["coverage"], "frozen": ctx["cal"]["frozen"],
            "dark_share": getattr(ctx["tm"], "dark_share", None), "strips": getattr(ctx["tm"], "strips", None), "width": ctx["vi"]["width"], "height": ctx["vi"]["height"]}
    os.makedirs(os.path.dirname(out), exist_ok=True); pickle.dump(keep, open(out, "wb"))
    log(f"{pid}: {ctx['vi']['n']} frames, {sum(len(v) for v in ctx['per'].values()) / max(1, len(ctx['per'])):.1f} players/frame")
    return out

def join(pieces, plan_, n_total, fps):
    """piece pickles -> one context for analyse(): global frame indices, unique tracker ids, every frame present"""
    per, H, cands = {}, {}, {}; cov = []; frozen = 0; meta = None
    for p, pl in zip(pieces, plan_):
        off = pl["offset"]; bump = pl["i"] * ID_STRIDE
        for k, rows in p["per"].items():
            g = off + k
            if g >= n_total: continue
            per[g] = [[r[0] + bump if r[0] >= 0 else r[0]] + list(r[1:]) for r in rows]
        for k, h in p["H"].items():
            if off + k < n_total: H[off + k] = h
        for k, c in p["cands"].items():
            if off + k < n_total: cands[off + k] = c
        cov.append(p["coverage"]); frozen += p["frozen"]; meta = meta or p
    known = sorted(H)
    import bisect
    for g in range(n_total):                                  # frames lost at a cut: no players, nearest calibration
        per.setdefault(g, []); cands.setdefault(g, [])
        if g not in H:
            j = bisect.bisect_left(known, g); cand = [known[x] for x in (j - 1, j) if 0 <= x < len(known)]
            H[g] = H[min(cand, key=lambda v: abs(v - g))]
    return per, H, cands, {"coverage": float(np.mean(cov)) if cov else 0.0, "frozen": frozen, "L": meta["L"], "W": meta["W"],
                           "width": meta["width"], "height": meta["height"], "dark_share": meta["dark_share"], "strips": meta["strips"]}

def remap_gt(src_gt, start_s, fps, dst):
    """labels of a segment cut at start_s -> the same frames in the full match"""
    gt = json.load(open(src_gt)); off = int(round(start_s * fps))
    json.dump({str(off + int(k)): v for k, v in gt.items()}, open(dst, "w")); return dst


def canary_index(plan_, todo_ids):
    """a piece from the middle of the first half (not warm-up), among those still to run"""
    if not todo_ids: return None
    target = round(0.2 * (len(plan_) - 1))
    return min(todo_ids, key=lambda i: abs(i - target))

def canary_ok(log_lines, expect_panorama, players_range=(6.0, 30.0)):   # panorama clips show every player at once (18-25); follow-cam pieces 6-16
    """(ok, reason) for the first piece of a full match, from its own log"""
    import re
    text = "\n".join(log_lines)
    if "PIECE FAILED" in text: return False, "the piece crashed"
    if "mosaic calibration failed" in text: return False, "the panorama calibration crashed"
    if expect_panorama and "calibration: from panorama" not in text: return False, "the panorama calibration was not used"
    m = re.search(r"(\d+) frames, ([\d.]+) players/frame", text)
    if not m: return False, "no player count in the piece's log"
    p = float(m.group(2))
    if not (players_range[0] <= p <= players_range[1]): return False, f"{p} players per frame is outside {players_range[0]:.0f}-{players_range[1]:.0f}"
    return True, f"calibration from the panorama, {p} players per frame"


def apply_periods(per, H, cands, periods_s, fps, L, W):
    """Keep only match time and normalise direction.
    periods_s: [(start_s, end_s), ...] in video seconds. Frames outside every period are blanked (no players, no ball) but
    stay on the timeline. Teams swap ends at half-time, so every period after the first is mirrored (x -> L-x, y -> W-y)
    THROUGH the calibration: H' = H @ M. Positions computed with H' come out mirrored, and drawing mirrored positions with
    H' lands on the same pixels, so overlays stay correct. Returns per, H, cands, play mask, period records."""
    from .tracking import reposition
    import numpy as np
    n = len(per); M = np.array([[-1.0, 0, L], [0, -1.0, W], [0, 0, 1.0]])
    spans = [(int(round(a * fps)), min(n, int(round(b * fps)))) for a, b in periods_s]
    which = np.full(n, -1, int)
    for idx, (a, b) in enumerate(spans): which[max(0, a):b] = idx
    H2 = {}; per2 = {}; cands2 = {}; mirrored = {}
    for k in range(n):
        p = which[k]
        if p < 0:
            per2[k] = []; cands2[k] = []; H2[k] = H[k]; continue
        cands2[k] = cands.get(k, [])
        if p >= 1: H2[k] = (H[k] @ M) if hasattr(H[k], "to_m") else np.asarray(H[k], float) @ M; mirrored[k] = per.get(k, [])
        else: H2[k] = H[k]; per2[k] = per.get(k, [])
    per2.update(reposition(mirrored, H2))
    records = [{"index": i + 1, "t_start": round(a / fps, 2), "t_end": round(b / fps, 2), "mirrored": i >= 1} for i, (a, b) in enumerate(spans)]
    return per2, H2, cands2, which >= 0, records


def apply_unknown(per, cands, ok):
    """frames whose calibration isn't trusted: no positions, no ball (like half-time); they stay on the timeline"""
    for k in range(len(ok)):
        if not ok[k]: per[k] = []; cands[k] = []
    return per, cands
