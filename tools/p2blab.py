"""P2b (4 Oct): Reymersholm whites vanish in tracking (results/qa/p2/README.md). The kit model learns only from people
the grass test keeps; at night it keeps 12-29% of them, the white team falls out of the kit groups and is removed as
'neither team'. Variants of the kit sample, fitted on each P2 piece's 8 key frames (raw frames + the detector's boxes,
results/qa/p2/<piece>/), then:
  - people per key frame read as team A / team B / neither (K) / off the pitch (O), like tracking reads them;
  - Reymersholm only: the same model graded on the 252 night players labelled by eye (results/qa/kitprobe, P7/P8).
Free, local: python tools/p2blab.py  -> results/qa/p2b/p2blab.json + sheets"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
P2 = "results/qa/p2"; OUT = "results/qa/p2b"; os.makedirs(OUT, exist_ok=True)
R = "results/qa/kitprobe"; RG = "p15u-vs-reymersholm-2026-09-18"
items = [tuple(x) for x in json.load(open(f"{R}/reym_label_items.json"))]
labels = json.load(open(f"{R}/reym_labels.json"))["labels_by_eye"]; P7L = json.load(open(f"{R}/reym_labels_p7.json"))
KEY = [(tuple(it), labels[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7L["items"], P7L["labels"])]
KB = json.load(open(f"{R}/{RG}/boxes.json")); KF = {fn: cv2.imread(f"{R}/{RG}/{fn}") for fn in KB}

def grade(pred):
    """G/W team names are arbitrary: G = whichever of A/B most green players get"""
    pairs = [(l, p) for (_, l), p in zip(KEY, pred)]
    a = "A" if sum(l == "G" and p == "A" for l, p in pairs) >= sum(l == "G" and p == "B" for l, p in pairs) else "B"; b = "B" if a == "A" else "A"
    nG = sum(l == "G" for l, _ in pairs); nW = sum(l == "W" for l, _ in pairs); nO = sum(l == "O" for l, _ in pairs)
    return {"team right": f"{sum((l == 'G' and p == a) or (l == 'W' and p == b) for l, p in pairs)}/{nG + nW}",
            "whites in white team": f"{sum(l == 'W' and p == b for l, p in pairs)}/{nW}", "whites neither/off": f"{sum(l == 'W' and p not in ('A', 'B') for l, p in pairs)}/{nW}",
            "greens in green team": f"{sum(l == 'G' and p == a for l, p in pairs)}/{nG}", "others kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{nO}"}

VARIANTS = {"now (grass test)": {"IPANEMA_PITCH_FALLBACK": "0"}, "fallback to edge test": {"IPANEMA_PITCH_FALLBACK": "0.35"},
            "edge test always": {"IPANEMA_PITCH_TEST": "edge", "IPANEMA_PITCH_FALLBACK": "0"}, "everyone (no pitch test)": {"_pitch_only": False}}

def fit(fb, v):
    old = {k: os.environ.get(k) for k in ("IPANEMA_PITCH_TEST", "IPANEMA_PITCH_FALLBACK")}
    for k in old: os.environ.pop(k, None)
    os.environ.update({k: x for k, x in v.items() if not k.startswith("_")}); logs = []
    try: m = K.KitTeamModel().fit_frames(fb, log=logs.append, pitch_only=v.get("_pitch_only", True))
    finally:
        for k, x in old.items():
            os.environ.pop(k, None)
            if x is not None: os.environ[k] = x
    m.offpitch = True; return m, logs

def sheet(crops, name, per_row=16):
    if not crops: return
    cs = [cv2.resize(c, (40, 80), interpolation=cv2.INTER_NEAREST) for c in crops]
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == "__main__":
    res = {}
    for d in sorted(glob.glob(f"{P2}/*{os.environ.get('P2B_ONLY', '')}*/")):
        piece = d.rstrip("/").split("/")[-1]; kd = json.load(open(d + "keydets.json"))
        fb = [(cv2.imread(f"{d}raw_{int(j):04d}.jpg"), np.array(bs)) for j, bs in kd.items() if bs]
        res[piece] = {}
        for vn, v in VARIANTS.items():
            m, logs = fit(fb, v); preds = [m.predict_batch(f, b) for f, b in fb]
            per = {t: float(np.median([p.count(t) for p in preds])) for t in "ABKO"}
            r = {"per frame A/B/neither/off": [per[t] for t in "ABKO"], "log": [l for l in logs if "learned from" in l][0][6:]}
            if RG in piece:
                r["252 labelled night players"] = grade([m.predict_batch(KF[fn], np.array([KB[fn][j]]))[0] for (fn, j), _ in KEY])
                if vn in ("now (grass test)", "fallback to edge test"):
                    for t in "AB": cv2.imwrite(f"{OUT}/{piece}_{'now' if vn.startswith('now') else 'fix'}_team{t}.jpg", cv2.resize(m.strips[t], (576, 144 * (m.strips[t].shape[0] // 72))))
                    sheet([f[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for (f, bs), p in zip(fb, preds) for b, t in zip(bs, p) if t == "K"][:96], f"{OUT}/{piece}_{'now' if vn.startswith('now') else 'fix'}_neither.jpg")
            res[piece][vn] = r; print(piece, "|", vn, "|", json.dumps({k: x for k, x in r.items() if k != "log"}), flush=True)
    json.dump(res, open(f"{OUT}/p2blab{os.environ.get('P2B_TAG', '')}.json", "w"), indent=1, ensure_ascii=False)
