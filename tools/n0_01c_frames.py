"""N0-01c (free runner, $0, nothing on Modal): blind film strips for 'who has the ball after P plays it?'
(results/qa/n0-01c/moments.json from tools/n0_01c_moments.py; the key is in moments_key.json, not read here).
Per moment one sheet: 6 frames evenly from t0 to t1, 3 x 2; the passer is ringed 'P' in the first frame only.
Run 1: one fixed crop box (640x360 tiles) - the Veo camera pans, so the play slid out of it. Run 2 (moments with 'tl'):
a 720x405 box per frame centred on the exported ball, shown 1:1.

    R2_PUBLIC_URL=... python tools/n0_01c_frames.py          # free runner
    SRC=some.mp4 python tools/n0_01c_frames.py              # dry run: one local video stands in for every match
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("IN", "results/qa/n0-01c/moments.json")
OUT = os.environ.get("OUT", "results/free/n0_01c")
TW, TH, N = 640, 360, 6
CW, CH = 720, 405          # run 2: per-frame box of this size at m['tl'][i] (centred on the exported ball), shown 1:1


def times(m, n=N):
    return [m["t0"] + (m["t1"] - m["t0"]) * i / (n - 1) for i in range(n)]


def tile(cv, f, m, i, dt):
    if f.shape[:2] != (1080, 1920): f = cv.resize(f, (1920, 1080))
    if m.get("tl") and m["tl"][i]: (x0, y0), w, h = m["tl"][i], CW, CH
    else: x0, y0, w, h = m["box"]
    tw = CW if m.get("tl") else TW; im = cv.resize(f[y0:y0 + h, x0:x0 + w], (tw, int(tw * h / w)), interpolation=cv.INTER_CUBIC); s = tw / w
    if i == 0:
        p = (int((m["passer_px"][0] - x0) * s), int((m["passer_px"][1] - y0) * s))
        cv.circle(im, p, int(18 * s) + 6, (0, 0, 255), 2, cv.LINE_AA)
        for c, wd in (((0, 0, 0), 5), ((0, 0, 255), 2)): cv.putText(im, "P", (p[0] + 14, p[1] - 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, c, wd)
    for c, wd in (((0, 0, 0), 5), ((255, 255, 255), 2)): cv.putText(im, f"{m['id']}  +{dt:.1f} s", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, c, wd)
    return im


def sheet(cv, np, frames, m):
    ims = [tile(cv, f, m, i, t - m["t0"]) for i, (t, f) in enumerate(frames)]
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
        for m in sorted((m for m in moments if m["match"] == mid), key=lambda m: m["t0"]):
            frs, ok_all = [], True
            for t in times(m):
                cap.set(cv.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
                if not ok: f = np.zeros((1080, 1920, 3), np.uint8); ok_all = False
                frs.append((t, f))
            name = f"{m['id']}.jpg"; cv.imwrite(os.path.join(OUT, name), sheet(cv, np, frs, m), [cv.IMWRITE_JPEG_QUALITY, 88])
            done.append({"id": m["id"], "file": name, "read_ok": ok_all}); print(name, mid, m["t0"], ok_all, flush=True)
        cap.release()
        if src.startswith("/tmp/") and not os.environ.get("SRC"):
            try: os.remove(src)
            except OSError: pass
    json.dump({"sheets": done}, open(os.path.join(OUT, "sheets.json"), "w"), indent=1)
    print(f"{len(done)} sheets")


if __name__ == "__main__":
    main()
