"""T1 (2 Oct, free runner, CPU): run the 3-frame ball scorers on EVERY guess the app's picker got (the exact app inputs,
results/volume/cache/<clip>/picker_inputs.pkl) for both clips, so tools/t1lab.py can test 'remove doubted guesses before the
picker' offline. Models: results/ball/t1/models/*.pt (tools/t1_train.py, leak-free for SFK-BP) + the Kaggle scorer_big.
Also a picture sheet per clip at the answer-key moments: every guess, the scorer's p, green frame = the real ball.
    R2_PUBLIC_URL=... python tools/t1_score.py            -> results/free/t1/<clip>.npz + sheet_<clip>.jpg
    LOCAL_CLIP=fake.mp4 MAX_FRAMES=60 python tools/t1_score.py     (dry run)"""
import sys, os, json, glob, time, pickle, subprocess, urllib.request, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    import torch
torch.set_num_threads(os.cpu_count() or 2)
from ipanema import ballscorer as SC, ballprobe as BP, ballfilter as BF
OUT = os.environ.get("T1_OUT", "results/free/t1"); os.makedirs(OUT, exist_ok=True)
CLIPS = os.environ.get("CLIPS", "p15u-vs-aik-2026-09-21-bd09_s2520,SFKBP1109_s1200").split(",")
MAXF = int(os.environ.get("MAX_FRAMES", "0"))
KEYS = {"p15u-vs-aik-2026-09-21-bd09_s2520": "reference/p15u-vs-aik-2026-09-21-bd09_s2520/ball_gt.json",
        "SFKBP1109_s1200": "results/volume/reference/SFKBP1109_s1200/ball_gt.json"}
paths = sorted(glob.glob("results/ball/t1/models/*.pt")) + ["results/kaggle/scorer_big/scorer_all_seed0.pt"]
nets = {}
for p in paths:
    net = SC.model(); net.load_state_dict(torch.load(p, map_location="cpu")); nets[os.path.basename(p)[:-3]] = net.eval()
print("models:", list(nets), flush=True)


def fetch(clip):
    if os.environ.get("LOCAL_CLIP"): return os.environ["LOCAL_CLIP"]
    dst = f"/tmp/t1_{clip}.mp4"
    if os.path.exists(dst): return dst
    req = urllib.request.Request(os.environ["R2_PUBLIC_URL"].rstrip("/") + f"/{clip}/video.mp4", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=900) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst


def tile(f, x, y, p, good, half=24, scale=2):
    pad = cv2.copyMakeBorder(f, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    xi, yi = int(round(x)) + half, int(round(y)) + half
    t = cv2.resize(pad[yi - half:yi + half, xi - half:xi + half], (2 * half * scale, 2 * half * scale), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (0, 220, 0) if good else (60, 60, 60), 3 if good else 1)
    cv2.putText(t, f"{p:.2f}", (3, t.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return t


for clip in CLIPS:
    t0 = time.time(); P = pickle.load(open(f"results/volume/cache/{clip}/picker_inputs.pkl", "rb")); cands = P["cands"]
    gt = {int(k): v for k, v in json.load(open(KEYS[clip])).items() if v}
    src = fetch(clip); print(clip, "video ready", round(time.time() - t0), "s", flush=True)
    cap = cv2.VideoCapture(src); n = len(cands) if not MAXF else min(MAXF, len(cands))
    frames = []; scores = {k: {} for k in nets}; rows_sheet = []; buf = {}

    def read():
        ok, f = cap.read()
        if not ok: return None
        return f if f.shape[1] == 1920 else cv2.resize(f, (1920, 1080))
    prev, cur, nxt = None, read(), read(); X, idx = [], []

    def flush():
        global X, idx
        if not X: return
        A = np.stack(X)
        for k, net in nets.items():
            p = SC.score(net, A)
            for (f, j), pi in zip(idx, p): scores[k].setdefault(f, {})[j] = float(pi)
        X, idx = [], []
    for i in range(n):
        if cur is None: break
        f3 = (prev if prev is not None else cur, cur, nxt if nxt is not None else cur)
        for j, (x, y, s) in enumerate(cands.get(i, [])):
            X.append(BP.crop3(f3, x, y, 32)); idx.append((i, j))
        if i in gt: buf[i] = cur.copy()
        if len(X) >= 2048: flush()
        prev, cur, nxt = cur, nxt, read()
        if i % 1000 == 0: print(f"  {clip}: frame {i}/{n}, {round(time.time() - t0)} s", flush=True)
    flush(); cap.release()
    done = sorted(scores[next(iter(nets))])
    arrays = {}
    for k in nets:
        fr, p = BF.pack({f: [scores[k][f][j] for j in range(len(cands[f]))] for f in done}); arrays[k] = p
    np.savez_compressed(f"{OUT}/{clip}.npz", fr=fr, models=np.array(list(nets)), **arrays)
    # picture sheet: every guess at each answer-key moment, p of the first model, green = within 30 px of the real ball
    k0 = next(iter(nets)); lines = []
    for f in sorted(buf):
        g = gt[f]; ts = []
        for j, (x, y, s) in sorted(enumerate(cands.get(f, [])), key=lambda z: -z[1][2])[:12]:
            ts.append(tile(buf[f], x, y, scores[k0][f][j], np.hypot(x - g[0], y - g[1]) <= 30))
        while len(ts) < 12: ts.append(np.zeros((96, 96, 3), np.uint8))
        lab = np.zeros((96, 70, 3), np.uint8); cv2.putText(lab, str(f), (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        lines.append(np.hstack([lab] + ts))
    if lines: cv2.imwrite(f"{OUT}/sheet_{clip}.jpg", np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump({"clip": clip, "frames": len(done), "guesses": int(len(fr)), "models": list(nets), "sheet_model": k0,
               "seconds": round(time.time() - t0)}, open(f"{OUT}/{clip}.json", "w"), indent=1)
    print(clip, "done", len(done), "frames", len(fr), "guesses", round(time.time() - t0), "s", flush=True)
