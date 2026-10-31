"""N0-31e (free runner, $0, nothing on Modal): blind picture sheets of final-third entries - the ones the new off-pitch rule
drops mixed with as many it keeps (results/qa/n0-31e/moments.json, made by tools/n0_31e_clips.py; key in moments_key.json,
not read here). Per entry one sheet: top row = 640x360 crops around the exported ball (red ring) at t-1 s, t-0.5 s and the
entry frame t; bottom row = the same full frames scaled down. Question by eye: at t, is the ring on the ball, and is the
ball on the pitch (not in the air / out / a spare ball)?

    R2_PUBLIC_URL=... python tools/n0_31e_frames.py          # free runner
    SRC=some.mp4 python tools/n0_31e_frames.py              # dry run: one local video stands in for every match
"""
import os, sys, json, bisect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("IN", "results/qa/n0-31e/moments.json")
OUT = os.environ.get("OUT", "results/free/n0_31e")
TW, TH = 640, 360


def ball_at(fr, ts, t):
    """exported ball pixels nearest to t (within 0.2 s) -> (t_frame, px) or (t, None)"""
    i = bisect.bisect_left(ts, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t), default=None)
    if best is None or abs(ts[best] - t) > 0.2: return t, None
    b = fr[best].get("ball"); return ts[best], (b or {}).get("px")


def plan(moments, loader):
    """moments -> [{id, match, tiles: [(label, t, px)]}] (the three times per entry)"""
    cache = {}; out = []
    for m in moments:
        if m["match"] not in cache:
            d, fr = loader(m["match"]); cache[m["match"]] = (fr, [f["t"] for f in fr])
        fr, ts = cache[m["match"]]; tiles = []
        for lab, dt in (("t-1.0", -1.0), ("t-0.5", -0.5), ("t", 0.0)):
            tt, px = ball_at(fr, ts, m["t"] + dt); tiles.append((lab, tt, px))
        out.append({"id": m["id"], "match": m["match"], "team": m["team"], "attacks": m["attacks"], "tiles": tiles})
    return out


def crop(cv, f, px, label):
    H0, W0 = f.shape[:2]
    if (W0, H0) != (1920, 1080): f = cv.resize(f, (1920, 1080))
    full = f.copy()
    if px is not None: cv.circle(full, (int(px[0]), int(px[1])), 22, (0, 0, 255), 3, cv.LINE_AA)
    cx = int(min(max((px or [960, 540])[0], TW / 2), 1920 - TW / 2)); cy = int(min(max((px or [960, 540])[1], TH / 2), 1080 - TH / 2))
    x0, y0 = cx - TW // 2, cy - TH // 2; im = f[y0:y0 + TH, x0:x0 + TW].copy()
    if px is not None: cv.circle(im, (int(px[0] - x0), int(px[1] - y0)), 14, (0, 0, 255), 2, cv.LINE_AA)
    for c, w in (((0, 0, 0), 4), ((255, 255, 255), 2)): cv.putText(im, label, (8, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7, c, w)
    return im, cv.resize(full, (TW, TH))


def main():
    import cv2 as cv, numpy as np
    from n0_30d_clips import load
    moments = json.load(open(IN)); os.makedirs(OUT, exist_ok=True)
    P = plan(moments, lambda mid: load(f"results/volume/runs/matches/{mid}/match_data.json"))
    from clipopen import open_clip
    done = []
    for mid in sorted({p["match"] for p in P}):
        src = os.environ.get("SRC")
        if not src:
            base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
            if not base: raise SystemExit("set R2_PUBLIC_URL or SRC")
            from clipopen import local_copy
            src = local_copy(f"{base}/{mid}/video.mp4", dest=f"/tmp/{mid}.mp4", max_bytes=12e9)
        cap, fps, n_all, _ = open_clip(src, cv)
        if cap is None: print(f"could not open {src}"); continue
        for p in [p for p in P if p["match"] == mid]:
            tiles = []
            for lab, t, px in p["tiles"]:
                cap.set(cv.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
                if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
                tiles.append(crop(cv, f, px, f"{p['id']} {lab} ({t:.1f} s)"))
            sheet = np.vstack([np.hstack([a for a, _ in tiles]), np.hstack([b for _, b in tiles])])
            name = f"{p['id']}.jpg"; cv.imwrite(os.path.join(OUT, name), sheet, [cv.IMWRITE_JPEG_QUALITY, 85])
            done.append(dict(p, file=name)); print(name, mid, flush=True)
        cap.release()
        if src.startswith("/tmp/") and not os.environ.get("SRC"):
            try: os.remove(src)
            except OSError: pass
    json.dump({"sheets": done}, open(os.path.join(OUT, "sheets.json"), "w"), indent=1)
    print(f"{len(done)} sheets")


if __name__ == "__main__":
    main()
