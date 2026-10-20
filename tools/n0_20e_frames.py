"""N0-20e (20 Oct, free runner): picture sheets of the app's two test clips at each detected restart
(results/qa/n0-20e/moments.json, built by tools/n0_20e_restarts.py), to grade by eye what the restart is and where the
ball really is when play restarts. Each sheet: tiles every 0.5 s from 4 s before the stoppage's end to 3 s after, 640 px,
time printed, the app's ball pick as a red ring, two candidate restart spots as crosses '1' (cyan) and '2' (magenta).
Which window gave which spot is NOT on the sheet (moments_key.json, opened after grading).
    R2_PUBLIC_URL=... python tools/n0_20e_frames.py      (or LOCAL_CLIP=clip.mp4 for a dry run: every clip -> that file)
    N0_20E_ZOOM=800: tiles are 800 x 450 crops (1920-px units) around the two spots instead of the whole picture"""
import os, sys, json, urllib.request, numpy as np, cv2
_a = sys.argv[1:]                                          # free runner: 'tools/n0_20e_frames.py --zoom 800' on the trigger line
if "--zoom" in _a: os.environ["N0_20E_ZOOM"] = _a[_a.index("--zoom") + 1]; os.environ.setdefault("N0_20E_OUT", "results/free/n0-20e/zoom")
M = json.load(open(os.environ.get("N0_20E_MOMENTS", "results/qa/n0-20e/moments.json")))
OUT = os.environ.get("N0_20E_OUT", "results/free/n0-20e"); TW, COLS = 640, 3; TH = TW * 9 // 16
COL = [(255, 255, 0), (255, 0, 255)]
ZOOM = int(os.environ.get("N0_20E_ZOOM", "0") or 0)      # >0: each tile is a ZOOM x ZOOM*9/16 crop (1920-px units) around the two spots
os.makedirs(OUT, exist_ok=True)


def fetch(clip):
    if os.environ.get("LOCAL_CLIP"): return os.environ["LOCAL_CLIP"]
    dst = f"/tmp/n0_20e_{clip}.mp4"
    if os.path.exists(dst): return dst
    url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{M['clips'][clip]}"
    print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


def put(im, txt, y, col=(255, 255, 255)):
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def tile(fr, tl, mid, s):
    ox = oy = 0.0
    if ZOOM:                                                                   # crop around the spots (and ball), in 1920 x 1080 units
        pts = [p for p in (tl.get("spot_px") or []) if p is not None] or ([tl["ball_px"]] if tl.get("ball_px") else [[960, 540]])
        cx, cy = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])); zw, zh = ZOOM, ZOOM * 9 // 16
        ox = min(max(cx - zw / 2, 0), 1920 - zw); oy = min(max(cy - zh / 2, 0), 1080 - zh); f = fr.shape[1] / 1920.0
        fr = fr[int(oy * f):int((oy + zh) * f), int(ox * f):int((ox + zw) * f)]; s = TW / float(zw)
    im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
    for i, p in enumerate(tl.get("spot_px") or []):
        if p is None: continue
        x, y = int((p[0] - ox) * s), int((p[1] - oy) * s)
        if -50 < x < TW + 50 and -50 < y < TH + 50:
            cv2.drawMarker(im, (x, y), COL[i], cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA); cv2.putText(im, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL[i], 2)
    if tl.get("ball_px"): cv2.circle(im, (int((tl["ball_px"][0] - ox) * s), int((tl["ball_px"][1] - oy) * s)), 9, (0, 0, 255), 1, cv2.LINE_AA)
    put(im, f"{mid}  t {tl['t']:.1f} s  ({tl['dt']:+.1f} s from stoppage end)", 22)
    return im


def main():
    made = {}; caps = {}
    for m in M["moments"]:
        if m["clip"] not in caps:
            cap = cv2.VideoCapture(fetch(m["clip"])); caps[m["clip"]] = (cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            print(f"{m['clip']}: {caps[m['clip']][1]} frames, {caps[m['clip']][2]:.0f} px wide", flush=True)
        cap, n, W0 = caps[m["clip"]]; s = TW / 1920.0      # spots/ball are in 1920x1080 pixels whatever the clip's own size
        tiles = []
        for tl in m["tiles"]:
            if tl["frame"] >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, tl["frame"]); ok, fr = cap.read()
            if not ok: break
            tiles.append(tile(fr, tl, m["id"], s))
        if not tiles: continue
        real = len(tiles)
        while len(tiles) % COLS: tiles.append(np.zeros((TH, TW, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)])
        cv2.imwrite(f"{OUT}/{m['id']}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); made[m["id"]] = real
    json.dump({"tile_w": TW, "sheets": made}, open(f"{OUT}/sheets.json", "w"), indent=1)
    print("sheets:", made, flush=True)


if __name__ == "__main__": main()
