"""N0-23a (free runner): picture sheets of the wins whose 'forward pass within 3 s' answer changed (or looked too quick),
from results/qa/n0-23a/moments.json (made locally by tools/n0_23a_clips.py). Each tile: 960x540 around the picked ball
(yellow ring), shown at 640x360, labelled with seconds since the loser's last control. No old/new labels (graded blind).
    R2_PUBLIC_URL=... python tools/n0_23a_sheet.py
    dry run: LOCAL_VIDEO=a.mp4 python tools/n0_23a_sheet.py <out_dir>"""
import os, sys, json, urllib.request
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "results/qa/n0-23a/moments.json")))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results/qa/n0-23a")


def fetch(clip):
    if os.environ.get("LOCAL_VIDEO"): return os.environ["LOCAL_VIDEO"]
    dst = f"/tmp/n0_23a_{clip}.mp4"
    if not os.path.exists(dst):
        url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"; print("downloading", clip, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})   # R2 refuses Python's default agent (403)
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b)
    return dst


def tile(img, bp, label, last):
    H, W = img.shape[:2]; c = bp or last or (W / 2, H / 2)
    x0 = int(min(max(0, c[0] - 480), max(0, W - 960))); y0 = int(min(max(0, c[1] - 270), max(0, H - 540)))
    t = img[y0:y0 + 540, x0:x0 + 960].copy()
    if bp: cv2.circle(t, (int(bp[0] - x0), int(bp[1] - y0)), 14, (0, 255, 255), 2)
    t = cv2.resize(t, (640, 360)); cv2.rectangle(t, (0, 0), (150, 34), (0, 0, 0), -1)
    cv2.putText(t, label, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2); return t


def main():
    os.makedirs(OUT, exist_ok=True); by_clip = {}
    for m in M: by_clip.setdefault(m["clip"], []).append(m)
    for clip, ms in by_clip.items():
        cap = cv2.VideoCapture(fetch(clip)); need = sorted({t["frame"] for m in ms for t in m["tiles"]}); got = {}; k = 0
        while need and k <= need[-1]:
            ok, f = cap.read()
            if not ok: break
            if k in need: got[k] = f
            k += 1
        for m in ms:
            tiles, last = [], None
            for t in m["tiles"]:
                f = got.get(t["frame"])
                if f is None: continue
                tiles.append(tile(f, t["ball_px"], f"{m['id']} {t['dt']:+.1f}s", last)); last = t["ball_px"] or last
            if not tiles: print("no frames for", m["id"]); continue
            while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(os.path.join(OUT, f"sheet_{m['id']}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", m["id"], len(m["tiles"]), "tiles", flush=True)


if __name__ == "__main__":
    main()
