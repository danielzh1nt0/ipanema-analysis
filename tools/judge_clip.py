"""29 Sep: score every WASB guess (30/frame) on the SFK-BP clip with the ball judge trained on Kaggle
(results/kaggle/scorer_big/scorer_all_seed0.pt). Free runner, CPU: reads the clip from R2 once, front to back.
-> results/ball/scorer/judge_clip_scores.npz (frame, x, y, finder score, judge score)."""
import os, sys, pickle, time, subprocess, numpy as np, cv2
try: import torch
except ImportError: subprocess.run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu', shell=True, check=True); import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballscorer as SC, ballprobe as BP
C = "results/volume/cache/SFKBP1109_s1200/"
o = pickle.load(open(C + "ball_cands_wasb_1790008894_t2x2_thr0.05_lp30.pkl", "rb")); o = o[0] if isinstance(o, tuple) else o
net = SC.model(); net.load_state_dict(torch.load("results/kaggle/scorer_big/scorer_all_seed0.pt", map_location="cpu")); net.eval()
src = os.environ.get("LOCAL_CLIP") or os.environ["R2_PUBLIC_URL"].rstrip("/") + "/SFKBP1109_s1200/video.mp4"
cap = cv2.VideoCapture(src); buf = []; k = -1; rows = []; t0 = time.time(); X = []; META = []
def flush():
    global X, META
    if not X: return
    p = SC.score(net, np.stack(X))
    for (kk, x, y, s), pp in zip(META, p): rows.append((kk, x, y, s, float(pp)))
    X, META = [], []
while True:
    ok, f = cap.read()
    if not ok: break
    k += 1; buf.append(f); buf = buf[-3:]
    j = k - 1                                                   # the middle frame of the last three
    if j < 0 or len(buf) < 3: continue
    for x, y, s in o.get(j, []):
        X.append(BP.crop3(tuple(buf), x, y, 32)); META.append((j, float(x), float(y), float(s)))
    if len(X) >= 4096: flush()
    if k % 1000 == 0: print(k, len(rows), f"{time.time() - t0:.0f}s", flush=True)
flush()
os.makedirs("results/ball/scorer", exist_ok=True)
np.savez_compressed("results/ball/scorer/judge_clip_scores.npz", rows=np.array(rows, np.float32))
print("done", len(rows), "guesses scored", f"{time.time() - t0:.0f}s")
