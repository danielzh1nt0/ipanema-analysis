# Kaggle (free, 2 Oct, S8): what are the 'not a ball' spots? 64 crops per clip from results/ball/negatives_2026-10-02.json
# (keyed-elsewhere and still-nobody-near), 96x96 around the spot, with the conf. For an eye check before any training.
import os, sys, json, subprocess, random, cv2, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True)
neg = json.load(open(f"{T}/ia/results/ball/negatives_2026-10-02.json")); random.seed(0)
for clip in ("p15u-vs-aik-2026-09-21-bd09_s2520", "SFKBP1109_s1200"):
    cap = cv2.VideoCapture(f"{R2}/{clip}/video.mp4")
    for why in ("keyed elsewhere", "still, nobody near"):
        rows = [r for r in neg if r["clip"] == clip and r["why"] == why]; random.shuffle(rows); rows = sorted(rows[:64], key=lambda r: r["frame"]); tiles = []
        for r in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"]); ok, f = cap.read()
            if not ok: continue
            f = cv2.resize(f, (1920, 1080)); x, y = int(r["x"]), int(r["y"]); x0, y0 = max(0, min(1920 - 96, x - 48)), max(0, min(1080 - 96, y - 48))
            t = cv2.resize(f[y0:y0 + 96, x0:x0 + 96].copy(), (192, 192)); cv2.putText(t, f"{r['conf']:.2f} f{r['frame']}", (2, 14), 0, 0.45, (0, 255, 255), 1); tiles.append(t)
        while len(tiles) % 8: tiles.append(np.zeros((192, 192, 3), np.uint8))
        rowsim = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        if rowsim: cv2.imwrite(f"{W}/{clip[:6]}_{why.split(',')[0].replace(' ', '_')}.jpg", np.vstack(rowsim), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(clip, why, len(tiles), flush=True)
json.dump({"ok": True}, open(f"{W}/result.json", "w"))
