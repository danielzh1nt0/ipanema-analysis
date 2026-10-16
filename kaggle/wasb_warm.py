# Kaggle (free GPU, 16 Oct, K8): K7 + WARM START from the app weights + the 100 volume SFK-BP labels. K7 numbers: Vallentuna 9->23, AIK 13->25, SFK-BP exam 18->15, clip 23->20 (close but still under on SFK-BP).
# K7 was: K6 but BALANCED (K6 regressed on SFK-BP: 59 clicks vs 1692 pseudo). WASB on all grounds: SFK-BP clicks + AIK clicks + Vallentuna pseudo-labels (K5, no human
# labels), from the pretrained soccer weights, compared with the weights the app runs today (models/wasb_finetuned.pth,
# fetched from the volume) on three honest tests. Original K5 notes:
# pseudo-labels from the export (tools/pseudo_ball.py, results/ball/pseudo/<match>.json), training exactly like
# ipanema.wasb_train.train (random tiles at the inference scale, 20% held out). Test, pretrained vs self-trained:
# the 55 by-eye Vallentuna ball keys (held out of the labels) and the 39 AIK clicks; top guess within 15 px, and within top 3.
import subprocess, os, sys, json, time, shutil
def sh(c): print("+", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2000:], flush=True); return r.returncode
t0 = time.time(); M = "p15u-vs-vallentuna-2026-10-03-6cce"; AIK = "p15u-vs-aik-2026-09-21-bd09_s2520"; R2 = "{{R2}}".rstrip("/"); W = "/kaggle/working"; T = "/kaggle/temp"
sh("nvidia-smi --query-gpu=name --format=csv"); sh(f"git clone -q --depth 1 https://github.com/danielzh1nt0/ipanema-analysis.git {T}/ia"); sh("pip install -q omegaconf gdown")
os.makedirs(f"{T}/root/reference/{M}", exist_ok=True); os.makedirs(f"{T}/videos", exist_ok=True); os.makedirs(f"{T}/root/models", exist_ok=True)
import random, cv2
# K7 balance: Vallentuna pseudo-labels capped to CAP (they are many near-duplicate frames); SFK-BP from Daniel's own click file
# (train split only; the exam quarter and anything inside the s1200 clip window stay out) with the "absent" frames as negatives.
CAP = int(os.environ.get("K7_CAP", "500"))
pseudo = json.load(open(f"{T}/ia/results/ball/pseudo/{M}.json")); ks = sorted(pseudo); random.Random(0).shuffle(ks)
json.dump({k: pseudo[k] for k in sorted(ks[:CAP], key=int)}, open(f"{T}/root/reference/{M}/ball_gt.json", "w"))
SFK = "SFKBP1109"; os.makedirs(f"{T}/root/reference/{SFK}", exist_ok=True)
sh(f"wget -q -O {T}/videos/{SFK}.mp4 {R2}/{SFK}/video.mp4; ls -la {T}/videos/{SFK}.mp4")
sfk_fps = cv2.VideoCapture(f"{T}/videos/{SFK}.mp4").get(cv2.CAP_PROP_FPS)
clicks = json.load(open(f"{T}/ia/results/labels/SFKBP1109_ball_clicks.json"))["frames"]
exam = [r for r in clicks if r["split"] == "exam"]; exam_t = [r["t"] for r in exam]
sfk_gt = {}
for r in clicks:
    if r["split"] != "train" or min(abs(r["t"] - t) for t in exam_t) < 1.0 or 1195 <= r["t"] <= 1505: continue
    if r["group"] in ("feet", "easy", "far") and r["x"] is not None: sfk_gt[int(round(r["t"] * sfk_fps))] = [r["x"], r["y"]]
    elif r["group"] == "absent": sfk_gt[int(round(r["t"] * sfk_fps))] = None
vol_gt = json.load(open(f"{T}/ia/results/volume/reference/{SFK}/ball_gt.json"))       # K8: the 100 volume labels too (not exam-adjacent, not the clip window)
for k, v in vol_gt.items():
    t = int(k) / sfk_fps
    if min(abs(t - x) for x in exam_t) >= 1.0 and not (1195 <= t <= 1505) and int(k) not in sfk_gt: sfk_gt[int(k)] = v
json.dump(sfk_gt, open(f"{T}/root/reference/{SFK}/ball_gt.json", "w"))
exam_gt = {int(round(r["t"] * sfk_fps)): [r["x"], r["y"]] for r in exam if r["group"] in ("feet", "easy", "far") and r["x"] is not None}
log0 = lambda *a: print(*a, flush=True); log0(f"K7: fps {sfk_fps}, SFK-BP train labels {len(sfk_gt)} ({sum(1 for v in sfk_gt.values() if v)} with ball), exam {len(exam_gt)}, Vallentuna pseudo {min(CAP, len(ks))} of {len(ks)}")
# the AIK clip clicks train too (its video is the eval clip; WT.EVAL_SETS keeps SFKBP1109_s1200 out but not this one)
os.makedirs(f"{T}/root/reference/{AIK}", exist_ok=True); shutil.copy(f"{T}/ia/reference/{AIK}/ball_gt.json", f"{T}/root/reference/{AIK}/ball_gt.json")
os.makedirs(f"{T}/root/reference/SFKBP1109_s1200", exist_ok=True)
if os.path.exists(f"{T}/ia/results/volume/reference/SFKBP1109_s1200/ball_gt.json"): shutil.copy(f"{T}/ia/results/volume/reference/SFKBP1109_s1200/ball_gt.json", f"{T}/root/reference/SFKBP1109_s1200/ball_gt.json")
sh(f"wget -q -O {T}/videos/SFKBP1109_s1200.mp4 {R2}/SFKBP1109_s1200/video.mp4; ls -la {T}/videos/SFKBP1109_s1200.mp4")
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
tests = {"Vallentuna by-eye keys (held out)": samples(f"{T}/videos/{M}.mp4", {int(k): v for k, v in json.load(open(f"{T}/ia/reference/{M}/ball_key_px.json")).items()}),
         "AIK clicks (in training, 80%)": samples(f"{T}/videos/{AIK}.mp4", {int(k): v for k, v in json.load(open(f"{T}/ia/reference/{AIK}/ball_gt.json")).items() if v}),
         "SFK-BP exam clicks (held out)": samples(f"{T}/videos/{SFK}.mp4", exam_gt)}
if os.path.exists(f"{T}/root/reference/SFKBP1109_s1200/ball_gt.json"):
    tests["SFK-BP clip moments (held out)"] = samples(f"{T}/videos/SFKBP1109_s1200.mp4", {int(k): v for k, v in json.load(open(f"{T}/root/reference/SFKBP1109_s1200/ball_gt.json")).items() if v})
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
app_w = f"{T}/ia/results/volume/models/wasb_finetuned.pth"
if os.path.exists(app_w):
    shutil.copy(app_w, f"{T}/root/models/wasb_finetuned.pth"); rep["app today (clicks fine-tune)"] = grade(WB._model(f"{T}/root", dev, finetuned=True), dev); log("app today", rep["app today (clicks fine-tune)"])
    os.remove(f"{T}/root/models/wasb_finetuned.pth")
init = f"{T}/root/models/wasb_app.pth" if os.path.exists(app_w) else None
if init: shutil.copy(app_w, init)
path = WT.train_best_of(f"{T}/root", f"{T}/videos", seeds=(0, 1), log=log, init=init)
if path:
    net = WB._model(f"{T}/root", dev, finetuned=True); rep["self-trained"] = grade(net, dev); log("self-trained", rep["self-trained"])
    shutil.copy(path, f"{W}/wasb_warm.pth")
rep["minutes"] = round((time.time() - t0) / 60, 1); json.dump(rep, open(f"{W}/result.json", "w"), indent=1, default=int); print(rep, flush=True)
