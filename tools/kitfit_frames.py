"""P6 (29 Sep): how many frames the kit model needs - fit on 6/12/24 random saved Reymersholm frames, graded on 93 people labelled by eye."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, "/home/claude/ipanema-analysis"); os.chdir("/home/claude/ipanema-analysis")
from ipanema import kits as K
g = "p15u-vs-reymersholm-2026-09-18"; D = f"results/qa/kitprobe/{g}"; B = json.load(open(f"{D}/boxes.json")); fns = sorted(B)
frames = {fn: cv2.imread(f"{D}/{fn}") for fn in fns}
items = [tuple(x) for x in json.load(open("results/qa/kitprobe/reym_label_items.json"))]
labels = json.load(open("results/qa/kitprobe/reym_labels.json"))["labels_by_eye"]
def score(m):
    pred = [m.predict_batch(frames[fn], [B[fn][j]])[0] for fn, j in items]
    pairs = [(labels[str(i)], p) for i, p in enumerate(pred)]
    gA = sum(1 for l, p in pairs if l == "G" and p == "A"); green = "A" if gA >= sum(1 for l, p in pairs if l == "G" and p == "B") else "B"; white = "B" if green == "A" else "A"
    ok = sum(1 for l, p in pairs if (l == "G" and p == green) or (l == "W" and p == white)); wA = sum(1 for l, p in pairs if l == "W" and p == green)
    return ok, wA
rng = np.random.default_rng(1)
for m_ in (6, 12, 24):
    res = []
    for t in range(12 if m_ < 24 else 1):
        sub = sorted(rng.choice(len(fns), m_, replace=False)) if m_ < 24 else range(24)
        m = K.KitTeamModel().fit_frames([(frames[fns[i]], np.array(B[fns[i]])) for i in sub], log=lambda *a: None)
        res.append(score(m) + (m.model["sizes"],))
    oks = [r[0] for r in res]; print(f"fit on {m_} frames: team right {min(oks)}-{max(oks)} of 71 (median {int(np.median(oks))}); whites called green up to {max(r[1] for r in res)}/31; e.g. group sizes {res[0][2]}")
