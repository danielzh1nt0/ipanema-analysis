"""6 Oct (P-PASS): picture strips of the model's pass moments in a pass clip, for labelling the passing team by eye.
Each row: t-0.4, t-0.2, t, t+0.3 s, centre crop 1100x620 of the 1280x720 clip (scaled). 4 rows per sheet.
    python tools/pass_moment_sheets.py <clip> <out_dir> [thr]"""
import sys, json, cv2, numpy as np
clip, out = sys.argv[1], sys.argv[2]; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
d = json.load(open(f"results/kaggle/tdeed_passes/tdeed/{clip}.json"))
P = sorted(e["frame"] / d["fps"] for e in d["predictions"] if e["label"] in {"PASS", "HIGH PASS", "CROSS", "FREE KICK"} and e["confidence"] >= thr)
Q = []
for t in P:
    if not Q or t - Q[-1] > 0.8: Q.append(t)
cap = cv2.VideoCapture(f"results/kaggle/pass_clips/{clip}.mp4"); fps = cap.get(5); rows = []
for n, q in enumerate(Q):
    tiles = []
    for dt in (-0.4, -0.2, 0.0, 0.3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((q + dt) * fps))); ok, f = cap.read()
        f = cv2.resize(f, (1920, 1080))[230:850, 410:1510] if ok else np.zeros((620, 1100, 3), np.uint8)
        f = cv2.resize(f, (550, 310)); cv2.putText(f, f"#{n} {q + dt:.1f}", (6, 24), 0, 0.7, (255, 255, 255), 2); tiles.append(f)
    rows.append(np.hstack(tiles))
for s in range(0, len(rows), 4):
    q = rows[s:s + 4]
    while len(q) < 4: q.append(np.zeros_like(rows[0]))
    cv2.imwrite(f"{out}/{clip}_{s // 4:02d}.jpg", np.vstack(q), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump([round(q, 2) for q in Q], open(f"{out}/{clip}_times.json", "w")); print(len(Q), "moments")
