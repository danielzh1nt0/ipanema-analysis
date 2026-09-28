"""E1 (28 Sep): picture strips for a who-has-the-ball answer key on OUR footage (SFK-BP clip), graded by Claude by eye.
Per moment: 5 frames (-0.8 s .. +0.8 s), a 480x270 native-resolution crop around our ball pick (2x zoom), players marked
(A = red dot, B = blue dot), our ball pick = yellow ring. Free runner (clip from R2). -> results/qa/who/strip_*.jpg"""
import os, json, cv2, numpy as np
D = json.load(open(os.environ.get("WHO", "results/review/who_moments.json"))); OUT = os.environ.get("WHO_OUT", "results/qa/who"); os.makedirs(OUT, exist_ok=True)
src = os.environ.get("LOCAL_CLIP") or os.environ["R2_PUBLIC_URL"].rstrip("/") + "/SFKBP1109_s1200/video.mp4"
cap = cv2.VideoCapture(src); print("frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), flush=True)
tiles_all = []
for j, m in enumerate(D["moments"]):
    k = m["frame"]; c = D["ball"].get(str(k))
    if c is None:
        pts = [r[2] for r in D["players"].get(str(k), []) if r[2]]; c = np.mean(pts, 0).tolist() if pts else [960, 540]
    x0 = int(min(max(c[0] - 240, 0), 1920 - 480)); y0 = int(min(max(c[1] - 135, 0), 1080 - 270)); row = []
    for d in range(-24, 25, 12):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k + d); ok, f = cap.read()
        if not ok: f = np.zeros((1080, 1920, 3), np.uint8)
        for pid, tm, px, fl in D["players"].get(str(k + d), []):
            if px: cv2.circle(f, (int(px[0]), int(px[1]) + 6), 5, (0, 0, 255) if tm == "A" else (255, 0, 0) if tm == "B" else (200, 200, 200), -1)
        b = D["ball"].get(str(k + d))
        if b: cv2.circle(f, (int(b[0]), int(b[1])), 14, (0, 255, 255), 2)
        t = f[y0:y0 + 270, x0:x0 + 480].copy(); cv2.putText(t, f"{d / 30:+.1f}s", (5, 262), 0, 0.6, (255, 255, 255), 2); row.append(t)
    strip = np.hstack(row); cv2.putText(strip, f"#{j} t={m['t']:.1f}s ours={m['ours']}", (5, 24), 0, 0.8, (255, 255, 255), 2)
    tiles_all.append(strip)
for s in range(0, len(tiles_all), 4):
    cv2.imwrite(f"{OUT}/strip_{s // 4:02d}.jpg", np.vstack(tiles_all[s:s + 4]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("sheets", (len(tiles_all) + 3) // 4)
