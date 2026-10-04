"""P2c (4 Oct): Reymersholm piece 4227 after P2b - greens read as the white team, spectators by the right-hand fence
tracked as players (results/qa/p2b/README.md). Kit model variants fitted on the piece's 8 key frames (raw frames + the
detector's boxes, results/qa/p2/<piece>/) and graded on the 88 people labelled by eye (results/qa/p2c/labels_4227.json).
Also run on the other 17 P2 pieces (no labels there): people per frame A/B/neither/off must not move.
Kitprobe grounds (fitted on their own key frames, like P8 / F1c): Reymersholm 252 and Spånga 279 labelled players,
SFK-BP: people that change. Reymersholm P2 pieces' models are also graded on the 252 (like tools/p2blab.py).
Free, local: python tools/p2clab.py  -> results/qa/p2c/p2clab.json + sheets"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
P2 = "results/qa/p2"; OUT = "results/qa/p2c"; os.makedirs(OUT, exist_ok=True)
PIECE = "p15u-vs-reymersholm-2026-09-18_4227"
LAB = json.load(open(f"{OUT}/labels_4227.json"))["labels"]
VARIANTS = {"before (P2b)": {"IPANEMA_CLS_MAX_FLIP": "0"}, "now (classifier self-check)": {}, "no player classifier": {"IPANEMA_KIT_CLS": "0"}}
SHORT = {"before (P2b)": "before", "now (classifier self-check)": "now", "no player classifier": "noclassifier"}
ENVS = ("IPANEMA_KIT_CLS", "IPANEMA_CLS_MAX_FLIP", "IPANEMA_PITCH_FALLBACK")

def run(fb, v):
    old = {k: os.environ.get(k) for k in ENVS}
    for k in old: os.environ.pop(k, None)
    os.environ.update(v); logs = []
    try:
        m = K.KitTeamModel().fit_frames(fb, log=logs.append); m.offpitch = True
        preds = [p for f, b in fb for p in m.predict_batch(f, b)]
    finally:
        for k, x in old.items():
            os.environ.pop(k, None)
            if x is not None: os.environ[k] = x
    return m, logs, preds

def grade(preds):
    """team names are arbitrary: G = the letter most green players get. O = off the pitch reading"""
    pairs = [(l, p) for l, p in zip(LAB, preds) if l != "?"]
    g = "A" if sum(l == "G" and p == "A" for l, p in pairs) >= sum(l == "G" and p == "B" for l, p in pairs) else "B"; w = "B" if g == "A" else "A"
    n = lambda c: sum(l == c for l, _ in pairs)
    return {"team right": f"{sum((l == 'G' and p == g) or (l == 'W' and p == w) for l, p in pairs)}/{n('G') + n('W')}",
            "greens in green team": f"{sum(l == 'G' and p == g for l, p in pairs)}/{n('G')}", "greens in white team": f"{sum(l == 'G' and p == w for l, p in pairs)}/{n('G')}",
            "whites in white team": f"{sum(l == 'W' and p == w for l, p in pairs)}/{n('W')}", "whites in green team": f"{sum(l == 'W' and p == g for l, p in pairs)}/{n('W')}",
            "spectators/staff kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{n('O')}",
            "keepers/referee not a team": f"{sum(l == 'K' and p not in ('A', 'B') for l, p in pairs)}/{n('K')}",
            "players dropped as off the pitch": f"{sum(l in 'GW' and p == 'O' for l, p in pairs)}/{n('G') + n('W')}"}

def sheet(crops, name, per_row=16):
    if not crops: return
    cs = [cv2.resize(c, (40, 80), interpolation=cv2.INTER_NEAREST) for c in crops]
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == "__main__":
    res = {}
    for d in sorted(glob.glob(f"{P2}/*{os.environ.get('P2C_ONLY', '')}*/")):
        piece = d.rstrip("/").split("/")[-1]; kd = json.load(open(d + "keydets.json"))
        fb = [(cv2.imread(f"{d}raw_{int(j):04d}.jpg"), np.array(bs)) for j, bs in kd.items() if bs]
        res[piece] = {}
        for vn, v in VARIANTS.items():
            m, logs, preds = run(fb, v); i = 0; per = {t: [] for t in "ABKO"}
            for f, b in fb:
                p = preds[i:i + len(b)]; i += len(b)
                for t in "ABKO": per[t].append(p.count(t))
            r = {"per frame A/B/neither/off": [float(np.median(per[t])) for t in "ABKO"], "log": [l[6:] for l in logs]}
            if "reymersholm" in piece:
                import tools.p2blab as B
                r["252 labelled night players"] = B.grade([m.predict_batch(B.KF[fn], np.array([B.KB[fn][j]]))[0] for (fn, j), _ in B.KEY])["team right"]
            if piece == PIECE:
                r["88 labelled people"] = grade(preds); r["preds"] = "".join(preds)
                crops = [f[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] for f, bs in fb for bb in bs]
                for t in "AB": sheet([c for c, p in zip(crops, preds) if p == t], f"{OUT}/4227_{SHORT[vn]}_team{t}.jpg")
            res[piece][vn] = r; print(piece, "|", vn, "|", json.dumps({k: x for k, x in r.items() if k not in ("log", "preds")}), flush=True)
    import tools.f1clab as F
    for g, lf, t1, t2 in (("p15u-vs-reymersholm-2026-09-18", "reym_labels.json", "G", "W"), ("p15u-vs-spanga-2026-09-25", "spanga_labels_p8.json", "D", "S"), ("SFKBP1109_s1200", None, None, None)):
        if os.environ.get("P2C_ONLY"): break
        fr = F.load(f"{F.R}/{g}"); fb = [(f, b) for _, f, b in fr]; idx = {fn: i for i, (fn, _, _) in enumerate(fr)}; res[g] = {}; base = None
        for vn, v in VARIANTS.items():
            old = {k: os.environ.get(k) for k in ENVS}
            for k in old: os.environ.pop(k, None)
            os.environ.update(v); logs = []
            m = K.KitTeamModel().fit_frames(fb, log=logs.append); P = [m.predict_batch(f, b) if len(b) else [] for f, b in fb]
            for k, x in old.items():
                os.environ.pop(k, None)
                if x is not None: os.environ[k] = x
            flat = [x for p in P for x in p]; base = base or flat; r = {"people that change vs before": sum(a != b for a, b in zip(base, flat)), "log": [l[6:] for l in logs if "classifier" in l]}
            if lf:
                L = json.load(open(f"{F.R}/{lf}"))
                if "items" not in L:
                    items = json.load(open(f"{F.R}/reym_label_items.json")); l0 = L["labels_by_eye"]; P7 = json.load(open(f"{F.R}/reym_labels_p7.json"))
                    key = [(tuple(it), l0[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7["items"], P7["labels"])]
                else: key = [(tuple(it), l) for it, l in zip(L["items"], L["labels"])]
                r["labelled players"] = F.grade([P[idx[fn]][j] for (fn, j), _ in key], [l for _, l in key], t1, t2)
            res[g][vn] = r; print(g, "|", vn, "|", json.dumps({k: x for k, x in r.items() if k != "log"}), flush=True)
    json.dump(res, open(f"{OUT}/p2clab{os.environ.get('P2C_TAG', '')}.json", "w"), indent=1, ensure_ascii=False)
