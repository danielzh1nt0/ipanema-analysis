"""N0-25b (free runner): picture sheets of the people our team heat maps put on a touchline + mid-pitch controls
(tools/n0_25b_moments.py). Each sheet: 4 moments, each as the whole picture (640 px, ring = the person's feet) next to a
640x360 crop at full resolution around the person. Which ones are controls is NOT on the sheet (moments_key.json).
    R2_PUBLIC_URL=... python tools/n0_25b_frames.py      (LOCAL_CLIP_SFK / LOCAL_CLIP_AIK = local videos for a dry run)"""
import os, sys, json, urllib.request, numpy as np, cv2
TW = 640; TH = 360


def fetch(clip, dst):
    url = os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4"; print("downloading", clip, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


def put(im, txt, y):
    cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4); cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def sheets(short):
    M = json.load(open(f"results/qa/n0-25b/moments_{short}.json")); out = f"results/free/n0-25b/{short}"; os.makedirs(out, exist_ok=True)
    src = os.environ.get(f"LOCAL_CLIP_{short.upper()}") or fetch(M["clip"], f"/tmp/n0_25b_{short}.mp4")
    cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); made = {}
    for m in M["moments"]:
        rows = []
        for tl in m["tiles"]:
            if tl["frame"] >= n: continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, tl["frame"]); ok, fr = cap.read()
            if not ok: continue
            sx = fr.shape[1] / 1920.0; px, py = tl["ball_px"][0] * sx, tl["ball_px"][1] * sx
            full = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA); f = TW / fr.shape[1]
            cv2.circle(full, (int(px * f), int(py * f)), 12, (0, 0, 255), 2, cv2.LINE_AA)
            x0 = int(np.clip(px - TW / 2, 0, fr.shape[1] - TW)); y0 = int(np.clip(py - TH * 0.7, 0, fr.shape[0] - TH))
            crop = fr[y0:y0 + TH, x0:x0 + TW].copy(); cv2.circle(crop, (int(px - x0), int(py - y0)), 22, (0, 0, 255), 2, cv2.LINE_AA)
            put(full, f"{m['id']}  t {tl['t']:.1f} s", 22); rows.append(np.hstack([full, crop]))
        if rows: cv2.imwrite(f"{out}/{m['id']}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); made[m["id"]] = len(rows)
    json.dump({"clip": M["clip"], "frames": n, "sheets": made}, open(f"{out}/sheets.json", "w"), indent=1); print(short, "sheets:", len(made), flush=True)
    return len(made) == len(M["moments"])


if __name__ == "__main__":
    ok = all([sheets(s) for s in ("sfk", "aik")]); sys.exit(0 if ok else 1)
