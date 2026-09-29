"""P1 (29 Sep): Spånga striped kits read as 'neither team'. Tries torso colour readings (median / mean / trimmed mean) and
other-kit rules on the saved frames, graded on the by-eye keys: Spånga 279 players (spanga_labels_p8.json), Reymersholm
252 night players (reym_labels*.json); SFK (no key) = how many people change reading vs the default, with a sheet to check.
Free, local: python tools/p1lab.py"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
R = "results/qa/kitprobe"; OUT = "results/qa/p1"; os.makedirs(OUT, exist_ok=True)
items = [tuple(x) for x in json.load(open(f"{R}/reym_label_items.json"))]
labels = json.load(open(f"{R}/reym_labels.json"))["labels_by_eye"]; P7L = json.load(open(f"{R}/reym_labels_p7.json"))
REYM = [(tuple(it), labels[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7L["items"], P7L["labels"])]
SPL = json.load(open(f"{R}/spanga_labels_p8.json")); SPANGA = [(tuple(it), l) for it, l in zip(SPL["items"], SPL["labels"])]
KEYS = {"p15u-vs-reymersholm-2026-09-18": (REYM, "G", "W"), "p15u-vs-spanga-2026-09-25": (SPANGA, "D", "S")}
GROUNDS = ["p15u-vs-spanga-2026-09-25", "p15u-vs-reymersholm-2026-09-18", "SFKBP1109_s1200"]
def grade(pred, key, t1, t2):
    pairs = [(l, p) for (_, l), p in zip(key, pred)]
    a = "A" if sum(l == t1 and p == "A" for l, p in pairs) >= sum(l == t1 and p == "B" for l, p in pairs) else "B"; b = "B" if a == "A" else "A"
    n1 = sum(l == t1 for l, _ in pairs); n2 = sum(l == t2 for l, _ in pairs); nO = sum(l == "O" for l, _ in pairs)
    return {"right": sum((l == t1 and p == a) or (l == t2 and p == b) for l, p in pairs), "of": n1 + n2,
            f"{t1} neither": sum(l == t1 and p not in "AB" for l, p in pairs), f"{t2} neither": sum(l == t2 and p not in "AB" for l, p in pairs),
            "wrong team": sum((l == t1 and p == b) or (l == t2 and p == a) for l, p in pairs),
            "others kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{nO}"}
def sheet(crops, name, per_row=12):
    if not crops: return
    cs = [cv2.resize(c, (48, 96), interpolation=cv2.INTER_NEAREST) for c in crops[:96]]
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
DATA = {}
for g in GROUNDS:
    B = json.load(open(f"{R}/{g}/boxes.json")); DATA[g] = ({fn: cv2.imread(f"{R}/{g}/{fn}") for fn in sorted(B)}, B)
VARIANTS = json.loads(os.environ.get("P1_VARIANTS", "null")) or {
    "before P1 (auto off)": {"IPANEMA_KIT_AUTO": "0"}, "P1 auto choice (new default)": {}, "mean": {"IPANEMA_TORSO_STAT": "mean"}, "trimmed mean": {"IPANEMA_TORSO_STAT": "trim"},
    "per-team spread": {"IPANEMA_KIT_SPREAD": "team"}, "mean + per-team spread": {"IPANEMA_TORSO_STAT": "mean", "IPANEMA_KIT_SPREAD": "team"},
    "trim + per-team spread": {"IPANEMA_TORSO_STAT": "trim", "IPANEMA_KIT_SPREAD": "team"},
    "far + mean": {"IPANEMA_KIT_PAIR": "far", "IPANEMA_TORSO_STAT": "mean"}, "far + mean + per-team spread": {"IPANEMA_KIT_PAIR": "far", "IPANEMA_TORSO_STAT": "mean", "IPANEMA_KIT_SPREAD": "team"}}
for v, env in VARIANTS.items():
    if v != "P1 auto choice (new default)": env.setdefault("IPANEMA_KIT_AUTO", "0")      # fixed readings: no auto choice
res = {}; base = {}
for v, env in VARIANTS.items():
    old = {k: os.environ.get(k) for k in env}; os.environ.update(env); res[v] = {}
    try:
        for g in GROUNDS:
            fr, B = DATA[g]; fns = sorted(B)
            m = K.KitTeamModel().fit_frames([(fr[fn], np.array(B[fn])) for fn in fns], log=lambda *a: None)
            p = {fn: m.predict_batch(fr[fn], np.array(B[fn])) for fn in fns}; flat = [x for fn in fns for x in p[fn]]
            r = {"A/B/neither": [flat.count("A"), flat.count("B"), flat.count("K")], "classifier runs": len(m.cls), "reading": f"{m.pair}/{m.stat}", "neither-share tried": m.choice}
            if g in KEYS: key, t1, t2 = KEYS[g]; r.update(grade([p[fn][j] for (fn, j), _ in key], key, t1, t2))
            if v == "before P1 (auto off)": base[g] = p
            else:
                ch = [(fn, j) for fn in fns for j in range(len(B[fn])) if p[fn][j] != base[g][fn][j]]; r["changed vs now"] = len(ch)
                if g == "SFKBP1109_s1200":
                    crop = lambda fn, j: fr[fn][max(0, int(B[fn][j][1])):int(B[fn][j][3]), max(0, int(B[fn][j][0])):int(B[fn][j][2])]
                    sheet([crop(fn, j) for fn, j in ch], f"{OUT}/sfk_changed_{v.replace(' ', '_').replace('(', '').replace(')', '')}.jpg")
            res[v][g[:14]] = r
        print(v, json.dumps(res[v]), flush=True)
    finally:
        for k, x in old.items():
            if x is None: os.environ.pop(k, None)
            else: os.environ[k] = x
json.dump(res, open(f"{OUT}/p1lab.json", "w"), indent=1)
