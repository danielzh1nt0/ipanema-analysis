"""T1 (2 Oct, local CPU): train the 3-frame ball-vs-not scorers that tools/t1_score.py runs on every guess of the app inputs.
Leak-free for the SFK-BP clip: crops from full-match frames 35000-46000 (the s1200 clip is 35964-44956) and the clip34 split
are left out. AIK is never in training. Variants:  sfk = SFK-BP clicks only;  sfk_k1 = + 4 other matches (K1/K1b crops).
Also reports, per model, the old top-guess test on the clip34 crops (held out here).  -> results/ball/t1/models/*.pt"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballscorer as SC
import torch
OUT = "results/ball/t1/models"; os.makedirs(OUT, exist_ok=True)
d = np.load("results/volume/cache/ballcrops/SFKBP1109_crops.npz", allow_pickle=True); X, meta = d["X"], d["meta"].tolist()
split = np.array([m[1] for m in meta]); fr = np.array([int(m[2]) for m in meta]); y = np.array([int(m[6]) for m in meta])
inwin = (split != "clip34") & (fr >= 35000) & (fr <= 46000)
tr = (split != "clip34") & ~inwin
te = split == "clip34"; Mte = [meta[i] for i in np.where(te)[0]]
print(f"SFK crops {len(X)}: train {tr.sum()} ({y[tr].sum()} ball), left out in clip window {inwin.sum()}, clip34 test {te.sum()}", flush=True)
ext = [np.load(p, allow_pickle=True) for p in ("results/kaggle/ballcrops_k1/k1_crops.npz", "results/free/k1b/k1_crops.npz")]
Xk = np.concatenate([e["X"] for e in ext]); yk = np.concatenate([[int(m[6]) for m in e["meta"]] for e in ext])
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]; EP = int(os.environ.get("EPOCHS", "20"))
rep = {"train_sfk": int(tr.sum()), "train_k1": int(len(Xk)), "left_out_window": int(inwin.sum()), "models": {}}
rep["clip34 finder only"] = SC.top1_per_frame(Mte, np.zeros(te.sum()), 1.0)
for name, (Xt, yt) in {"sfk": (X[tr], y[tr]), "sfk_k1": (np.concatenate([X[tr], Xk]), np.concatenate([y[tr], yk]))}.items():
    for seed in SEEDS:
        net = SC.train(Xt, yt, epochs=EP, batch=64, seed=seed); p = SC.score(net, X[te])
        tag = f"{name}_s{seed}"; torch.save(net.state_dict(), f"{OUT}/{tag}.pt")
        rep["models"][tag] = {f"clip34 scorer + {w} x finder": SC.top1_per_frame(Mte, p, w) for w in (0.0, 0.5, 1.0)}
        print(tag, rep["models"][tag], flush=True)
json.dump(rep, open("results/ball/t1/train.json", "w"), indent=1)
