# Kaggle (free GPU, 16 Oct, K5): self-train the WASB ball finder on Vallentuna's own footage with NO human labels:
# pseudo-labels from the export (tools/pseudo_ball.py, results/ball/pseudo/<match>.json), training exactly like
# ipanema.wasb_train.train (random tiles at the inference scale, 20% held out). Test, pretrained vs self-trained:
# the 55 by-eye Vallentuna ball keys (held out of the labels) and the 39 AIK clicks; top guess within 15 px, and within top 3.
import subprocess, os, sys, json, time, shutil
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2000:], flush=True); return r.returncode
t0 = time.time(); M = "p15u-vs-vallentuna-2026-10-03-6cce"; AIK = "p15u-vs-aik-2026-09-21-bd09_s2520"; R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
sh("nvidia-smi --query-gpu=name --format=csv"); sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia"); sh("pip install -q omegaconf gdown")
os.makedirs(f"{T}/root/reference/{M}", exist_ok=True); os.makedirs(f"{T}/videos", exist_ok=True); os.makedirs(f"{T}/root/models", exist_ok=True)
shutil.copy(f"{T}/ia/results/ball/pseudo/{M}.json", f"{T}/root/reference/{M}/ball_gt.json")
sh(f"wget -q -O {T}/videos/{M}.mp4 {R2}/{M}/video.mp4; ls -la {T}/videos/{M}.mp4")
sh(f"wget -q -O {T}/videos/{AIK}.mp4 {R2}/{AIK}/video.mp4; ls -la {T}/videos/{AIK}.mp4")
sys.path.insert(0, f"{T}/ia"); os.environ["WASB_DIR"] = f"{T}/WASB-SBDT"
import numpy as np, cv2, torch
from ipanema import wasb as WB, wasb_train as WT
log = lambda *a: print(*a, flush=True)
# honest tests: by-eye Vallentuna keys (video frame -> px at 1920x1080) and the AIK clicks
def samples(video, gt):
    cap = cv2.VideoCapture(video); Wd, Hd = int(cap.get(3)), int(cap.get(4)); S = []
    for i, g in sorted(gt.items()):
        frames = []
        for j in (i - 1, i, i + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, j)); ok, f = cap.read()
            if not ok: break
            frames.append(WB.to_base(f))
        if len(frames) == 3: S.append({"frames": frames, "ball": (g[0] * WB.BASE[0] / Wd, g[1] * WB.BASE[1] / Hd)})
    return S
tests = {"Vallentuna by-eye keys": samples(f"{T}/videos/{M}.mp4", {int(k): v for k, v in json.load(open(f"{T}/ia/reference/{M}/ball_key_px.json")).items()}),
         "AIK clicks": samples(f"{T}/videos/{AIK}.mp4", {int(k): v for k, v in json.load(open(f"{T}/ia/reference/{AIK}/ball_gt.json")).items() if v})}
log({k: len(v) for k, v in tests.items()})
def grade(net, dev):
    out = {}
    for name, S in tests.items():
        top1 = top3 = 0
        for s in S:
            pk = WT.tiled_predict(net, dev, s["frames"]); d = [np.hypot(p[0] - s["ball"][0], p[1] - s["ball"][1]) for p in pk[:3]]
            top1 += bool(d) and d[0] <= 7.5; top3 += any(x <= 7.5 for x in d)
        out[name] = {"top1": top1, "top3": top3, "n": len(S)}
    return out
WB.ensure(f"{T}/root", log=log); dev = "cuda" if torch.cuda.is_available() else "cpu"
rep = {"pretrained": grade(WB._model(f"{T}/root", dev, finetuned=False), dev)}; log("pretrained", rep["pretrained"])
path = WT.train_best_of(f"{T}/root", f"{T}/videos", seeds=(0, 1), log=log)
if path:
    net = WB._model(f"{T}/root", dev, finetuned=True); rep["self-trained"] = grade(net, dev); log("self-trained", rep["self-trained"])
    shutil.copy(path, f"{W}/wasb_selftrained_{M}.pth")
rep["minutes"] = round((time.time() - t0) / 60, 1); json.dump(rep, open(f"{W}/result.json", "w"), indent=1); print(rep, flush=True)
