# Kaggle (free, 4 Oct): Vallentuna ball answer key, stage 1: our RF-DETR ball finder's top 3 guesses on each of the 40 random
# moments (results/review/vall/moments2.json), each as a 160-px zoom crop (x3) plus a small overview with the 3 guesses ringed.
# Claude grades A/B/C/none by eye -> reference/<match>/ball_gt.json. Leans to balls the finder saw (same caveat as the AIK key).
import os, sys, json, subprocess, cv2, urllib.request, numpy as np
R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
subprocess.run(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia", shell=True)
subprocess.run('pip install -q "rfdetr==1.11.0" supervision', shell=True); sys.path.insert(0, f"{T}/ia")
from ipanema import ballrf as BR
D = json.load(open(f"{T}/ia/results/review/vall/moments2.json")); cap = cv2.VideoCapture(f"{R2}/{D['src_key']}"); fps = cap.get(5)
ball = BR.load(f"{T}/ia/{BR.WEIGHTS}"); out = {}; R = 80
for j, t in enumerate(D["ball_moments"]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if not ok: continue
    f = cv2.resize(f, (1920, 1080)); c = BR.detect_many(ball, [f])[0][:3]; out[f"c{j:02d}"] = {"t": t, "guesses": [[round(float(x), 1), round(float(y), 1), round(float(cf), 3)] for x, y, cf in c]}
    tiles = []
    for k, (x, y, cf) in enumerate(c):
        x0, y0 = int(min(max(x - R, 0), 1920 - 2 * R)), int(min(max(y - R, 0), 1080 - 2 * R)); z = cv2.resize(f[y0:y0 + 2 * R, x0:x0 + 2 * R], (480, 480), interpolation=cv2.INTER_CUBIC)
        cv2.circle(z, (int((x - x0) * 3), int((y - y0) * 3)), 30, (0, 255, 255), 2); cv2.putText(z, f"{'ABC'[k]} {cf:.2f}", (6, 24), 0, 0.8, (0, 255, 255), 2); tiles.append(z)
    while len(tiles) < 3: tiles.append(np.zeros((480, 480, 3), np.uint8))
    ov = f.copy()
    for k, (x, y, cf) in enumerate(c): cv2.circle(ov, (int(x), int(y)), 22, (0, 255, 255), 3); cv2.putText(ov, "ABC"[k], (int(x) + 24, int(y)), 0, 1.2, (0, 255, 255), 3)
    ov = cv2.resize(ov, (854, 480)); cv2.putText(ov, f"c{j:02d} t={t:.1f}s", (8, 30), 0, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f"{W}/c{j:02d}.jpg", np.hstack(tiles + [ov]), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(j, flush=True)
json.dump(out, open(f"{W}/guesses.json", "w"), indent=1); json.dump({"ok": True, "n": len(out)}, open(f"{W}/result.json", "w")); print("done", len(out))
