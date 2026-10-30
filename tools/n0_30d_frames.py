"""N0-30d (free runner, $0, nothing on Modal): pictures of the Vallentuna 'won the ball' events whose 5-s gain changes with
the N0-30d rule (results/qa/n0-30d/clips_gain5s.json, made by tools/n0_30d_clips.py). Per event one sheet: top row = 640x360
crops around the ball pick (red ring) at the start, and at the two candidate end points (old: exactly +5 s; new: the last
on-pitch ball), bottom row = the full frames scaled down. The two ends are shown as 'end X' / 'end Y' in a seeded random
order (key in key.json) so they are graded blind: where is the ball really, and is the ring on it?

    R2_PUBLIC_URL=... python tools/n0_30d_frames.py          # free runner
    SRC=some.mp4 python tools/n0_30d_frames.py               # dry run on a local video
"""
import os, sys, json, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = "p15u-vs-vallentuna-2026-10-03-6cce"
IN = os.environ.get("IN", "results/qa/n0-30d/clips_gain5s.json")
OUT = os.environ.get("OUT", "results/free/n0_30d")
TW, TH = 640, 360


def picks(res, match=M, min_change=3.0):
    """changed events of one match -> [{i, t_won, start, X, Y}] with X/Y the two end points in a seeded random order"""
    rnd = random.Random(30); out = []
    for i, r in enumerate(res[match]["rows"]):
        if r.get("fps") is None or r["old_start"] is None: continue
        changed = (r["old"] is None) != (r["new"] is None) or (r["old"] is not None and abs(r["old"] - r["new"]) > min_change)
        if not changed: continue
        ends = [("old", r["old_end"], r["old"]), ("new", r["new_end"], r["new"])]
        if rnd.random() < 0.5: ends.reverse()
        out.append({"i": i, "t_won": r["t_won"], "team": r["team"], "start": r["new_start"] or r["old_start"],
                    "X": {"rule": ends[0][0], "pick": ends[0][1], "gain": ends[0][2]}, "Y": {"rule": ends[1][0], "pick": ends[1][1], "gain": ends[1][2]}})
    return out


def crop(cv, f, px, label):
    H0, W0 = f.shape[:2]
    if (W0, H0) != (1920, 1080): f = cv.resize(f, (1920, 1080))
    full = f.copy()
    if px is not None:
        cv.circle(full, (int(px[0]), int(px[1])), 22, (0, 0, 255), 3, cv.LINE_AA)
    cx = int(min(max((px or [960, 540])[0], TW / 2), 1920 - TW / 2)); cy = int(min(max((px or [960, 540])[1], TH / 2), 1080 - TH / 2))
    x0, y0 = cx - TW // 2, cy - TH // 2; im = f[y0:y0 + TH, x0:x0 + TW].copy()
    if px is not None: cv.circle(im, (int(px[0] - x0), int(px[1] - y0)), 14, (0, 0, 255), 2, cv.LINE_AA)
    for c, w in (((0, 0, 0), 4), ((255, 255, 255), 2)): cv.putText(im, label, (8, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, c, w)
    return im, cv.resize(full, (TW, TH))


def grab(cv, cap, fps, t):
    cap.set(cv.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    return f if ok else None


def main():
    import cv2 as cv, numpy as np
    res = json.load(open(IN)); P = picks(res); os.makedirs(OUT, exist_ok=True)
    src = os.environ.get("SRC")
    if not src:
        base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if not base: raise SystemExit("set R2_PUBLIC_URL or SRC")
        from clipopen import local_copy
        src = local_copy(f"{base}/{M}/video.mp4", dest="/tmp/vall_video.mp4", max_bytes=12e9)
    from clipopen import open_clip
    cap, fps, n_all, _ = open_clip(src, cv)
    if cap is None: raise SystemExit(f"could not open {src}")
    done = []
    for k, p in enumerate(P):
        tiles = []
        for lab, pk in (("start", p["start"]), ("end X", p["X"]["pick"]), ("end Y", p["Y"]["pick"])):
            f = grab(cv, cap, fps, pk["t"]) if pk else None
            if f is None: f = np.zeros((1080, 1920, 3), np.uint8)
            tiles.append(crop(cv, f, pk["px"] if pk else None, f"{lab} t={pk['t'] if pk else '-'}"))
        sheet = np.vstack([np.hstack([a for a, _ in tiles]), np.hstack([b for _, b in tiles])])
        name = f"g{k:02d}.jpg"; cv.imwrite(os.path.join(OUT, name), sheet, [cv.IMWRITE_JPEG_QUALITY, 85])
        done.append(dict(p, file=name)); print(name, p["t_won"], flush=True)
    cap.release()
    json.dump({"match": M, "fps": fps, "events": done}, open(os.path.join(OUT, "key.json"), "w"), indent=1)
    print(f"{len(done)} sheets")


if __name__ == "__main__":
    main()
