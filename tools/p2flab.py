"""P2f (9 Oct): Reymersholm piece 4227 at night - the dim far whites read as the green team (15/40 in the P2e re-track).
Their torso colour is cream (L 42-65, a* -4..-12, b* +10..+26), nearer the dim green kit centre than the bright white one,
but its HUE leans from the green kit's towards yellow. New rule (kits.dim_light_team, uncalibrated grounds only):
a coloured-kit reading whose hue leans >= IPANEMA_DIM_LIGHT degrees (default 18) towards yellow is the light kit.
Kit model fitted on each P2 piece's 8 key frames (like tools/p2clab.py), rule off vs on:
  - 4227: graded on the 88 people labelled by eye (results/qa/p2c/labels_4227.json);
  - Reymersholm pieces: also graded on the 252 labelled night players (kitprobe key, tools/p2blab.py);
  - every other piece: people that change (crops in changed_people.jpg, to grade by eye); kitprobe Reymersholm / Spånga
    labelled keys and SFK-BP key frames with offpitch on (tracktest setting).
Also a sweep of the degree threshold on 4227 + the 252 key.
Free, local: python tools/p2flab.py  -> results/qa/p2f/p2flab.json + sheets"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
P2 = "results/qa/p2"; OUT = "results/qa/p2f"; os.makedirs(OUT, exist_ok=True)
PIECE = "p15u-vs-reymersholm-2026-09-18_4227"
LAB = json.load(open("results/qa/p2c/labels_4227.json"))["labels"]

def run(fb, deg, extra=None):
    old = os.environ.get("IPANEMA_DIM_LIGHT"); os.environ["IPANEMA_DIM_LIGHT"] = str(deg)
    try:
        logs = []; m = K.KitTeamModel().fit_frames(fb, log=logs.append); m.offpitch = True
        preds = [p for f, b in fb for p in m.predict_batch(f, b)]
        ex = extra(m) if extra else None
    finally:
        os.environ.pop("IPANEMA_DIM_LIGHT", None)
        if old is not None: os.environ["IPANEMA_DIM_LIGHT"] = old
    return m, preds, ex

def grade(preds):
    import tools.p2clab as C
    C.LAB = LAB; return C.grade(preds)

def crop(f, b, tag):
    x1, y1, x2, y2 = [int(v) for v in b]; c = cv2.resize(f[max(0, y1):y2, max(0, x1):x2], (60, 120), interpolation=cv2.INTER_CUBIC)
    cv2.putText(c, tag, (2, 12), 0, 0.38, (0, 0, 255), 1); return c

def sheet(cs, name, per_row=12):
    if not cs: return
    cs = list(cs)
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 88])

if __name__ == "__main__":
    import tools.p2blab as B
    g252 = lambda m: B.grade([m.predict_batch(B.KF[fn], np.array([B.KB[fn][j]]))[0] for (fn, j), _ in B.KEY])["team right"]
    res = {"pieces": {}, "sweep_4227": {}}; changed = []
    for d in sorted(glob.glob(f"{P2}/*/")):
        piece = d.rstrip("/").split("/")[-1]; kd = json.load(open(d + "keydets.json"))
        fb = [(cv2.imread(f"{d}raw_{int(j):04d}.jpg"), np.array(bs)) for j, bs in kd.items() if bs]
        ex = g252 if "reymersholm" in piece else None
        m0, p0, e0 = run(fb, 0, ex); m1, p1, e1 = run(fb, K.DIM_LIGHT, ex)
        r = {"core hue (coloured kit index, deg)": [m1.core[0], round(m1.core[1], 1)] if m1.core else None,
             "per frame A/B/neither/off before": [float(np.median([p0[sum(len(b) for _, b in fb[:i]):][:len(fb[i][1])].count(t) for i in range(len(fb))])) for t in "ABKO"],
             "per frame A/B/neither/off now": [float(np.median([p1[sum(len(b) for _, b in fb[:i]):][:len(fb[i][1])].count(t) for i in range(len(fb))])) for t in "ABKO"],
             "people that change": sum(a != b for a, b in zip(p0, p1))}
        if ex: r["252 labelled night players before/now"] = [e0, e1]
        if piece == PIECE: r["88 labelled before"] = grade(p0); r["88 labelled now"] = grade(p1)
        i = 0
        for j, (f, bs) in enumerate(fb):
            for b in bs:
                if p0[i] != p1[i]: changed.append((piece, list(kd)[j], i, p0[i], p1[i], LAB[i] if piece == PIECE else "", crop(f, b, f"{len(changed)} {p0[i]}>{p1[i]}")))
                i += 1
        res["pieces"][piece] = r; print(piece, json.dumps(r), flush=True)
    sheet([c[-1] for c in changed], f"{OUT}/changed_people.jpg")
    res["changed"] = [{"n": n, "piece": c[0], "frame": c[1], "person": c[2], "before": c[3], "now": c[4], "label_4227": c[5]} for n, c in enumerate(changed)]
    d = f"{P2}/{PIECE}/"; kd = json.load(open(d + "keydets.json")); fb = [(cv2.imread(f"{d}raw_{int(j):04d}.jpg"), np.array(bs)) for j, bs in kd.items() if bs]
    for deg in (0, 10, 14, 16, 18, 20, 22, 26, 30):
        m, p, e = run(fb, deg, g252); g = grade(p)
        res["sweep_4227"][deg] = {"88: team right": g["team right"], "whites in green team": g["whites in green team"], "greens in white team": g["greens in white team"], "252 key": e}
        print("sweep", deg, res["sweep_4227"][deg], flush=True)
    # kitprobe keys (each ground's own key frames, tracktest setting offpitch on): Reymersholm 252, Spånga 279, SFK-BP changes
    import tools.f1clab as F
    res["kitprobe"] = {}
    for g, lf, t1, t2 in (("p15u-vs-reymersholm-2026-09-18", "reym_labels.json", "G", "W"), ("p15u-vs-spanga-2026-09-25", "spanga_labels_p8.json", "D", "S"), ("SFKBP1109_s1200", None, None, None)):
        fr = F.load(f"{F.R}/{g}"); fbk = [(f, b) for _, f, b in fr]; idx = {fn: i for i, (fn, _, _) in enumerate(fr)}; out = {}; base = None
        for deg in (0, K.DIM_LIGHT):
            os.environ["IPANEMA_DIM_LIGHT"] = str(deg)
            m = K.KitTeamModel().fit_frames(fbk, log=lambda *_: None); m.offpitch = True
            P = [m.predict_batch(f, b) if len(b) else [] for f, b in fbk]; flat = [x for p in P for x in p]; base = base or flat
            r = {"people that change vs off": sum(a != b for a, b in zip(base, flat)), "core": [m.core[0], round(m.core[1], 1)] if m.core else None}
            if lf:
                L = json.load(open(f"{F.R}/{lf}"))
                if "items" not in L:
                    items = json.load(open(f"{F.R}/reym_label_items.json")); l0 = L["labels_by_eye"]; P7 = json.load(open(f"{F.R}/reym_labels_p7.json"))
                    key = [(tuple(it), l0[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7["items"], P7["labels"])]
                else: key = [(tuple(it), l) for it, l in zip(L["items"], L["labels"])]
                r["labelled players"] = F.grade([P[idx[fn]][j] for (fn, j), _ in key], [l for _, l in key], t1, t2)
            out["off" if deg == 0 else "on"] = r
        os.environ.pop("IPANEMA_DIM_LIGHT", None)
        res["kitprobe"][g] = out; print(g, json.dumps(out), flush=True)
    json.dump(res, open(f"{OUT}/p2flab.json", "w"), indent=1, ensure_ascii=False)
