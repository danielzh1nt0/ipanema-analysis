"""N0-01b (free runner, $0, nothing on Modal): blind picture sheets for 'which team is the passer on?'
(results/qa/n0-01b/moments.json from tools/n0_01b_moments.py; the key is in moments_key.json, not read here).
Per pass one sheet: top row = 2x enlarged crops of the passer (red ring, 'P'), reference player '1' and reference '2'
(two players of different teams in the same frame); bottom = the whole frame scaled down with the three marked.
Question by eye: does P's shirt look like 1's, like 2's, or neither / can't tell?

    R2_PUBLIC_URL=... python tools/n0_01b_frames.py          # free runner
    SRC=some.mp4 python tools/n0_01b_frames.py              # dry run: one local video stands in for every match
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("IN", "results/qa/n0-01b/moments.json")
OUT = os.environ.get("OUT", "results/free/n0_01b")
CW, CH = 240, 200          # crop around a player's feet (px are feet points), enlarged 2x


def crop_box(px, W=1920, H=1080):
    """crop (x0, y0) so the player (feet at px) sits low-centre: most of the crop is above the feet"""
    x0 = int(min(max(px[0] - CW / 2, 0), W - CW)); y0 = int(min(max(px[1] - CH * 0.8, 0), H - CH)); return x0, y0


def tile(cv, f, px, label, colour):
    x0, y0 = crop_box(px); im = f[y0:y0 + CH, x0:x0 + CW].copy()
    cv.circle(im, (int(px[0] - x0), int(px[1] - y0)), 10, colour, 2, cv.LINE_AA)
    im = cv.resize(im, (CW * 2, CH * 2), interpolation=cv.INTER_CUBIC)
    for c, w in (((0, 0, 0), 5), ((255, 255, 255), 2)): cv.putText(im, label, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, c, w)
    return im


def sheet(cv, np, f, m):
    if f.shape[:2] != (1080, 1920): f = cv.resize(f, (1920, 1080))
    red, wht = (0, 0, 255), (255, 255, 255)
    top = np.hstack([tile(cv, f, m["passer_px"], f"P  {m['id']}", red), tile(cv, f, m["ref1_px"], "1", wht), tile(cv, f, m["ref2_px"], "2", wht)])
    full = f.copy()
    for px, lab, col in ((m["passer_px"], "P", red), (m["ref1_px"], "1", wht), (m["ref2_px"], "2", wht)):
        cv.circle(full, (int(px[0]), int(px[1])), 26, col, 4, cv.LINE_AA)
        cv.putText(full, lab, (int(px[0]) + 28, int(px[1]) - 28), cv.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
    full = cv.resize(full, (top.shape[1], int(top.shape[1] * 1080 / 1920)))
    return np.vstack([top, full])


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
            cap.set(cv.CAP_PROP_POS_FRAMES, int(round(m["t"] * fps))); ok, f = cap.read()
            if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
            name = f"{m['id']}.jpg"; cv.imwrite(os.path.join(OUT, name), sheet(cv, np, f, m), [cv.IMWRITE_JPEG_QUALITY, 85])
            done.append({"id": m["id"], "file": name, "read_ok": bool(ok)}); print(name, mid, m["t"], ok, flush=True)
        cap.release()
        if src.startswith("/tmp/") and not os.environ.get("SRC"):
            try: os.remove(src)
            except OSError: pass
    json.dump({"sheets": done}, open(os.path.join(OUT, "sheets.json"), "w"), indent=1)
    print(f"{len(done)} sheets")


if __name__ == "__main__":
    main()
