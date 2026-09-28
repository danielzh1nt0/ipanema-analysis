"""P5 (29 Sep): team colours at night. Fits the kit model on all on-pitch people of the 24 saved Reymersholm frames and
grades it on 93 people labelled by eye (results/qa/kitprobe/reym_labels.json), for several colour readings; also checks
Spånga and SFK by team balance. Free, local."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
def load(g):
    D = f"results/qa/kitprobe/{g}"; B = json.load(open(f"{D}/boxes.json"))
    return [(cv2.imread(f"{D}/{fn}"), np.array(B[fn])) for fn in sorted(B)], B
def run(g, lw, grass_d, items=None, labels=None):
    orig = K.torso_feature
    def tf(frame, box, grass=None, min_px=6, grass_d_=grass_d, green_kit=False):
        x1, y1, x2, y2 = [float(v) for v in box]; w, h = x2 - x1, y2 - y1
        c = frame[int(y1 + 0.20 * h):int(y1 + 0.48 * h), int(x1 + 0.30 * w):int(x1 + 0.70 * w) + 1]
        if c.size == 0: return None
        lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        gr = K.grass_lab(frame) if grass is None else grass
        lab = lab[np.linalg.norm(lab - gr, axis=1) > grass_d_]
        if len(lab) < min_px: return None
        L, a, b = np.median(lab, axis=0); return np.array([L / lw, a - 128.0, b - 128.0])
    K.torso_feature = tf
    try:
        fb, B = load(g); m = K.KitTeamModel().fit_frames(fb, log=lambda *a: None)
        if items is None:
            c = {"A": 0, "B": 0, "K": 0, None: 0}
            for f, b in fb:
                for x in m.predict_batch(f, b): c[x if x in c else "K"] += 1
            return c
        frames = {fn: cv2.imread(f"results/qa/kitprobe/{g}/{fn}") for fn in B}
        pred = [m.predict_batch(frames[fn], [B[fn][j]])[0] for fn, j in items]
        pairs = [(labels[str(i)], p) for i, p in enumerate(pred)]
        # which model team is green: majority over G-labelled
        gA = sum(1 for l, p in pairs if l == "G" and p == "A"); gB = sum(1 for l, p in pairs if l == "G" and p == "B")
        green = "A" if gA >= gB else "B"; white = "B" if green == "A" else "A"
        ok = sum(1 for l, p in pairs if (l == "G" and p == green) or (l == "W" and p == white))
        n = sum(1 for l, _ in pairs if l in "GW"); wrong_team = sum(1 for l, p in pairs if (l == "G" and p == white) or (l == "W" and p == green))
        other_ok = sum(1 for l, p in pairs if l == "O" and p not in ("A", "B")); no = sum(1 for l, _ in pairs if l == "O")
        return {"team right": f"{ok}/{n}", "wrong team": wrong_team, "other kept out": f"{other_ok}/{no}"}
    finally: K.torso_feature = orig
items = [tuple(x) for x in json.load(open("results/qa/kitprobe/reym_label_items.json"))]
labels = json.load(open("results/qa/kitprobe/reym_labels.json"))["labels_by_eye"]
res = {}
for lw in (2.5, 1.5, 1.0):
    for gd in (16.0, 24.0):
        r = run("p15u-vs-reymersholm-2026-09-18", lw, gd, items, labels)
        bal = {g: run(g, lw, gd) for g in ("p15u-vs-spanga-2026-09-25", "SFKBP1109_s1200")}
        res[f"L/{lw} grass_d {gd}"] = {"reymersholm (93 labelled)": r, "team balance spanga / sfk": bal}
        print(f"L/{lw} grass_d {gd}: {r} | spanga {bal['p15u-vs-spanga-2026-09-25']} | sfk {bal['SFKBP1109_s1200']}", flush=True)
json.dump(res, open("results/qa/kitprobe/kitlab_night.json", "w"), indent=1)
