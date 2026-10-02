# Kaggle (free, 2 Oct, F2b): AIK players missed in the big shadow. 10 frames from the AIK full match, RF-DETR medium as in the
# pipeline (conf 0.3) vs conf 0.15 vs conf 0.3 on a gamma-lifted frame (shadow brightened). Boxes drawn, one sheet per frame.
import os, sys, json, subprocess, cv2, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True); subprocess.run('pip install -q "rfdetr==1.11.0" supervision', shell=True)
sys.path.insert(0, f"{T}/ia"); os.environ["IPANEMA_DETECTOR"] = "rfdetr"
from ipanema import tracking as TR
det = TR.RFDetrPerson("medium"); cap = cv2.VideoCapture(f"{R2}/p15u-vs-aik-2026-09-21-bd09/video.mp4"); fps = cap.get(5)
def lift(f, gamma=0.6):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8); return cv2.LUT(f, lut)
rep = []
for j, t in enumerate([615, 1113, 2108, 2606, 3104, 4161, 4659, 5157, 5655, 6152]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (1920, 1080)); tiles = []; counts = {}
    for name, img, conf in (("conf 0.3 (pipeline)", f, 0.3), ("conf 0.15", f, 0.15), ("conf 0.3 + gamma 0.6", lift(f), 0.3)):
        d = det.detect_batch([img], conf, TR.FOLLOW_TILES)[0][0]; g = f.copy(); counts[name] = len(d.xyxy)
        for (a, b, c, e) in d.xyxy: cv2.rectangle(g, (int(a), int(b)), (int(c), int(e)), (0, 255, 255), 2)
        cv2.putText(g, f"t={t}s {name}: {len(d.xyxy)} people", (10, 32), 0, 1.0, (255, 255, 255), 2); tiles.append(cv2.resize(g, (1280, 720)))
    cv2.imwrite(f"{W}/m{j:02d}.jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82]); rep.append({"t": t, **counts}); print(rep[-1], flush=True)
json.dump({"ok": True, "rows": rep}, open(f"{W}/result.json", "w"), indent=1)
