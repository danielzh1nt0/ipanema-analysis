"""P8 (29 Sep): per-player team classifier (kits.fit_player_cls on body histograms) vs colour alone.
Grades on the 252 Reymersholm night players (results/qa/kitprobe/reym_labels*.json) and 279 Spånga players
(spanga_labels_p8.json), labelled by eye; for SFK (no labels) counts how many people change team. Sheets of the people
who change team, to check by eye.
Free, local: python tools/p8lab.py"""
import os, sys, json, time, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
R = "results/qa/kitprobe"; OUT = "results/qa/p8"; os.makedirs(OUT, exist_ok=True)
GROUNDS = ["p15u-vs-reymersholm-2026-09-18", "p15u-vs-spanga-2026-09-25", "SFKBP1109_s1200"]
def load(g):
    B = json.load(open(f"{R}/{g}/boxes.json")); return {fn: cv2.imread(f"{R}/{g}/{fn}") for fn in sorted(B)}, B
items = [tuple(x) for x in json.load(open(f"{R}/reym_label_items.json"))]
labels = json.load(open(f"{R}/reym_labels.json"))["labels_by_eye"]; P7L = json.load(open(f"{R}/reym_labels_p7.json"))
ALL = [(tuple(it), labels[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7L["items"], P7L["labels"])]
SPL = json.load(open(f"{R}/spanga_labels_p8.json")); SPANGA = [(tuple(it), l) for it, l in zip(SPL["items"], SPL["labels"])]
KEYS = {"p15u-vs-reymersholm-2026-09-18": (ALL, "G", "W"), "p15u-vs-spanga-2026-09-25": (SPANGA, "D", "S")}
def grade(pred, key=ALL, t1="G", t2="W"):
    """team names are arbitrary: team 1 = whichever of A/B most of its players get"""
    pairs = [(l, p) for (_, l), p in zip(key, pred)]
    a = "A" if sum(l == t1 and p == "A" for l, p in pairs) >= sum(l == t1 and p == "B" for l, p in pairs) else "B"; b = "B" if a == "A" else "A"
    n1 = sum(l == t1 for l, _ in pairs); n2 = sum(l == t2 for l, _ in pairs); nO = sum(l == "O" for l, _ in pairs)
    return {"team right": f"{sum((l == t1 and p == a) or (l == t2 and p == b) for l, p in pairs)}/{n1 + n2}",
            f"{t2} read as {t1}": f"{sum(l == t2 and p == a for l, p in pairs)}/{n2}", f"{t1} read as {t2}": f"{sum(l == t1 and p == b for l, p in pairs)}/{n1}",
            "others kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{nO}"}
def sheet(crops, name, per_row=12):
    if not crops: return
    cs = [cv2.resize(c, (48, 96), interpolation=cv2.INTER_NEAREST) for c in crops]
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
res = {}
for g in GROUNDS:
    fr, B = load(g); fns = sorted(B); fb = [(fr[fn], np.array(B[fn])) for fn in fns]
    for test in ("grass", "edge"):
        t0 = time.time()
        old = K.KitTeamModel().fit_frames(fb, log=lambda *a: None, pitch_test=test, player_cls=False)
        new = K.KitTeamModel().fit_frames(fb, log=lambda *a: None, pitch_test=test, player_cls=True)
        secs = round(time.time() - t0, 1)
        po = {fn: old.predict_batch(fr[fn], np.array(B[fn])) for fn in fns}; pn = {fn: new.predict_batch(fr[fn], np.array(B[fn])) for fn in fns}
        flat_o = [x for fn in fns for x in po[fn]]; flat_n = [x for fn in fns for x in pn[fn]]
        changed = [(fn, j) for fn in fns for j in range(len(B[fn])) if po[fn][j] != pn[fn][j]]
        r = {"people": len(flat_o), "A/B/K colour only": [flat_o.count(t) for t in "ABK"], "A/B/K with classifier": [flat_n.count(t) for t in "ABK"],
             "team changed": len(changed), "voting runs": len(new.cls), "seconds (both fits)": secs}
        if g in KEYS:
            key, t1, t2 = KEYS[g]
            r["colour only"] = grade([po[fn][j] for (fn, j), _ in key], key, t1, t2); r["with classifier"] = grade([pn[fn][j] for (fn, j), _ in key], key, t1, t2)
        crop = lambda fn, j: fr[fn][max(0, int(B[fn][j][1])):int(B[fn][j][3]), max(0, int(B[fn][j][0])):int(B[fn][j][2])]
        for a, b in (("A", "B"), ("B", "A")):          # people moved from a to b by the classifier
            sheet([crop(fn, j) for fn, j in changed if po[fn][j] == a and pn[fn][j] == b][:96], f"{OUT}/{g}_{test}_moved_{a}to{b}.jpg")
        cv2.imwrite(f"{OUT}/{g}_{test}_kits_new.jpg", np.vstack([cv2.resize(new.strips[t], (576, 144)) for t in ("A", "B")]))
        res[f"{test} | {g}"] = r; print(test, g, json.dumps(r), flush=True)
json.dump(res, open(f"{OUT}/p8lab.json", "w"), indent=1)
