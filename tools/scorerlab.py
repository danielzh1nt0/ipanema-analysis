"""T0 (28 Sep): train the ball-vs-not scorer on the TRAIN clicks' crops, grade on the exam (full SFK-BP match) and the 34
clip moments: per frame, is the best-scored guess the ball? Baseline = the finders' own score. CPU, free.
    python tools/scorerlab.py [crops.npz] [extra train crops.npz ...]
K1 (29 Sep): extra npz files (other matches, tools/k1_crops.py) are added to TRAINING only; MAXDK=n keeps followed balls
within n frames of Daniel's check (0 = checked frames only); TAG names the output (scorerlab_<TAG>.json)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ballscorer as SC
path = sys.argv[1] if len(sys.argv) > 1 else "results/volume/cache/ballcrops/SFKBP1109_crops.npz"
d = np.load(path, allow_pickle=True); X, meta = d["X"], d["meta"].tolist(); y = np.array([m[6] for m in meta])
for xp in sys.argv[2:]:                                               # K1: other matches, training only
    e = np.load(xp, allow_pickle=True); em = e["meta"].tolist(); keep = [i for i, m in enumerate(em) if abs(int(m[8])) <= int(os.environ.get("MAXDK", "99"))]
    X = np.concatenate([X, e["X"][keep]]); meta += [em[i] for i in keep]; y = np.concatenate([y, [em[i][6] for i in keep]])
split = np.array([m[1] for m in meta]); tr = split == "train"
TAG = os.environ.get("TAG", ""); OUTJ = f"results/ball/scorer/scorerlab{'_' + TAG if TAG else ''}.json"
print(f"{len(X)} crops: train {tr.sum()} ({y[tr].sum()} ball), exam {(split == 'exam').sum()}, clip {(split == 'clip34').sum()}", flush=True)
res = {}
for seed in (0, 1, 2):
    net = SC.train(X[tr], y[tr], epochs=20, batch=64, seed=seed); p = SC.score(net, X)
    for sp in ("exam", "clip34"):
        idx = np.where(split == sp)[0]; M = [meta[i] for i in idx]
        for fw in (0.0, 0.5, 1.0):
            res.setdefault(f"{sp} scorer + {fw} x finder", []).append(SC.top1_per_frame(M, p[idx], fw))
    if seed == 0 and not TAG:
        import torch; os.makedirs("results/ball/scorer", exist_ok=True); torch.save(net.state_dict(), "results/ball/scorer/scorer_seed0.pt")
for sp in ("exam", "clip34"):
    idx = np.where(split == sp)[0]; M = [meta[i] for i in idx]
    res[f"{sp} finder score only (baseline)"] = [SC.top1_per_frame(M, np.zeros(len(idx)), 1.0)]
for k, v in res.items(): print(f"{k:40s} " + "  ".join(f"{a}/{b}" for a, b in v), flush=True)
json.dump({k: [list(map(int, x)) for x in v] for k, v in res.items()}, open(OUTJ, "w"), indent=1)
