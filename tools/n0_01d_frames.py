"""N0-01d (free runner, $0, nothing on Modal): blind sheets for the export's spots (results/qa/n0-01d/moments.json from
tools/n0_01d_moments.py; which ones are off the pitch is in moments_key.json, not read here). Per moment 6 frames,
3 x 2, each a 720x405 box at 1:1 centred on the exported spot (cyan cross, drawn through that frame's own homography;
pitch outline in thin white). Frames without a homography are shown centred like the previous one, with no cross.

    R2_PUBLIC_URL=... python tools/n0_01d_frames.py          # free runner
    SRC=some.mp4 python tools/n0_01d_frames.py              # dry run: one local video stands in for every match
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("IN", "results/qa/n0-01d/moments.json")
OUT = os.environ.get("OUT", "results/free/n0_01d")
CW, CH = 720, 405


def project(H, x, y):
    if H is None: return None
    a, b, c, d, e, f, g, h, i = H; w = g * x + h * y + i
    if abs(w) < 1e-9: return None
    return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def box_for(pt, last):
    """top-left of a CW x CH box centred on pt (clamped into 1920x1080); no point -> the last box"""
    if pt is None: return last or [600, 340]
    x0 = int(min(max(pt[0] - CW / 2, 0), 1920 - CW)); y0 = int(min(max(pt[1] - CH / 2, 0), 1080 - CH))
    return [x0, y0]


def tile(cv, np, f, m, i, last):
    if f.shape[:2] != (1080, 1920): f = cv.resize(f, (1920, 1080))
    H = m["H"][i]; pt = project(H, *m["xy"]); tl = box_for(pt, last); x0, y0 = tl
    im = f.copy()
    if H is not None:
        L, W = m["L"], m["W"]; edge = []
        for (xa, ya), (xb, yb) in (((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0))):
            for s in range(41):
                q = project(H, xa + (xb - xa) * s / 40, ya + (yb - ya) * s / 40)
                edge.append(q)
            edge.append(None)
        prev = None
        for q in edge:
            if q is not None and prev is not None and max(abs(q[0]), abs(prev[0])) < 6000 and max(abs(q[1]), abs(prev[1])) < 6000:
                cv.line(im, (int(prev[0]), int(prev[1])), (int(q[0]), int(q[1])), (255, 255, 255), 1, cv.LINE_AA)
            prev = q
        if pt is not None and -50 < pt[0] < 1970 and -50 < pt[1] < 1130:
            p = (int(pt[0]), int(pt[1]))
            for c, wd in (((0, 0, 0), 4), ((255, 255, 0), 2)):
                cv.line(im, (p[0] - 14, p[1]), (p[0] + 14, p[1]), c, wd, cv.LINE_AA); cv.line(im, (p[0], p[1] - 14), (p[0], p[1] + 14), c, wd, cv.LINE_AA)
    im = im[y0:y0 + CH, x0:x0 + CW].copy()
    lab = f"{m['id']}  {m['ts'][i] - m['t']:+.1f} s" + ("" if H is not None else "  (no calibration)")
    for c, wd in (((0, 0, 0), 5), ((255, 255, 255), 2)): cv.putText(im, lab, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, c, wd)
    return im, tl


def sheet(cv, np, frames, m):
    ims, last = [], None
    for i, f in enumerate(frames):
        im, last = tile(cv, np, f, m, i, last); ims.append(im)
    return np.vstack([np.hstack(ims[:3]), np.hstack(ims[3:])])


def main():
    import cv2 as cv, numpy as np
    from clipopen import open_clip, local_copy
    moments = json.load(open(IN)); os.makedirs(OUT, exist_ok=True); done = []
    for mid in sorted({m["match"] for m in moments}):
        src = os.environ.get("SRC")
        if not src:
            base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
            if not base: raise SystemExit("set R2_PUBLIC_URL or SRC")
            src = local_copy(f"{base}/{mid}/video.mp4", dest=f"/tmp/{mid}.mp4", max_bytes=12e9)
        cap, fps, n_all, _ = open_clip(src, cv)
        if cap is None: print(f"could not open {src}"); continue
        for m in sorted((m for m in moments if m["match"] == mid), key=lambda m: m["t"]):
            frs, ok_all = [], True
            for t in m["ts"]:
                cap.set(cv.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
                if not ok: f = np.zeros((1080, 1920, 3), np.uint8); ok_all = False
                frs.append(f)
            name = f"{m['id']}.jpg"; cv.imwrite(os.path.join(OUT, name), sheet(cv, np, frs, m), [cv.IMWRITE_JPEG_QUALITY, 88])
            done.append({"id": m["id"], "file": name, "read_ok": ok_all}); print(name, mid, m["t"], ok_all, flush=True)
        cap.release()
        if src.startswith("/tmp/") and not os.environ.get("SRC"):
            try: os.remove(src)
            except OSError: pass
    json.dump({"sheets": done}, open(os.path.join(OUT, "sheets.json"), "w"), indent=1)
    print(f"{len(done)} sheets")


if __name__ == "__main__":
    main()
