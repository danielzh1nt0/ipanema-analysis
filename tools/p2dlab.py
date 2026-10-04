"""P2d (4 Oct): Reymersholm 4227 leftovers (results/qa/p2c/README.md). Side-touchline test (kits.side_lines): people whose
feet are beyond a long white side line, with only a small part of the pitch beyond it, read off the pitch ("O").
Kit model fitted once per piece on its 8 key frames (default settings), then the same model reads people with the side
test off (IPANEMA_SIDE_LINES=0, old) and on. Graded on:
  - 4227: the 88 people labelled by eye (results/qa/p2c/labels_4227.json, tools/p2clab.grade);
  - Reymersholm kitprobe frames: the 252 labelled night players + labelled spectators (tools/p2blab.grade);
  - every other P2 piece + Spånga / SFK-BP kitprobe frames: people that change, with a crop sheet of each one to check by eye.
Free, local: python tools/p2dlab.py -> results/qa/p2d/p2dlab.json + sheets"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
P2 = "results/qa/p2"; OUT = "results/qa/p2d"; os.makedirs(OUT, exist_ok=True)

def predict(m, fb, side):
    old = os.environ.get("IPANEMA_SIDE_LINES"); os.environ["IPANEMA_SIDE_LINES"] = "1" if side else "0"
    try: return [m.predict_batch(f, b) if len(b) else [] for f, b in fb]
    finally:
        os.environ.pop("IPANEMA_SIDE_LINES", None)
        if old is not None: os.environ["IPANEMA_SIDE_LINES"] = old

def crop(f, b, pad=0.6):
    x1, y1, x2, y2 = [int(v) for v in b[:4]]; w, h = x2 - x1, y2 - y1; H, W = f.shape[:2]
    c = f[max(0, int(y1 - pad * h)):min(H, int(y2 + pad * h)), max(0, int(x1 - 2 * w)):min(W, int(x2 + 2 * w))].copy()
    return cv2.resize(c, (120, 140))

def overlay(f, b_all, p_old, p_new, lines, name):
    o = f.copy()
    for a, b, c, y0, y1 in lines:                                   # draw the side line across its rows
        ys = np.array([y0, y1]); xs = -(b * ys + c) / (a if abs(a) > 1e-6 else 1e-6)
        cv2.line(o, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), (0, 0, 255), 4)
    for bb, po, pn in zip(b_all, p_old, p_new):
        x1, y1, x2, y2 = [int(v) for v in bb[:4]]
        col = (255, 0, 255) if (po != "O" and pn == "O") else ((128, 128, 128) if pn == "O" else (0, 255, 0))
        cv2.rectangle(o, (x1, y1), (x2, y2), col, 3)
    cv2.imwrite(name, cv2.resize(o, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])

def run_set(name, fb, fns, res, changed):
    m = K.KitTeamModel().fit_frames(fb, log=lambda *_: None); m.offpitch = True
    P0 = predict(m, fb, False); P1 = predict(m, fb, True)
    per = lambda P, t: float(np.median([p.count(t) for p in P])) if P else 0.0
    ch = []
    for (f, b), fn, p0, p1 in zip(fb, fns, P0, P1):
        lines = K.side_lines(f)
        for bb, a, c in zip(b, p0, p1):
            if a != c: ch.append((name, fn, a, c)); changed.append((name, fn, crop(f, bb), a, c))
        if any(a != c for a, c in zip(p0, p1)): overlay(f, b, p0, p1, lines, f"{OUT}/frame_{name[:40]}_{fn}.jpg")
    res[name] = {"per frame A/B/neither/off old": [per(P0, t) for t in "ABKO"], "new": [per(P1, t) for t in "ABKO"],
                 "people that change": len(ch), "changes": [f"{fn}: {a}->{c}" for _, fn, a, c in ch],
                 "frames with a side line": sum(bool(K.side_lines(f)) for f, _ in fb)}
    return m, P0, P1

def sheet(items, name, per_row=10):
    if not items: return
    cs = []
    for n, fn, c, a, b in items:
        c = c.copy(); cv2.putText(c, f"{a}>{b}", (3, 14), 0, 0.45, (0, 255, 255), 1); cv2.putText(c, f"{n.split('_')[-1][:8]} {fn}"[:20], (3, 135), 0, 0.35, (255, 255, 255), 1); cs.append(c)
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

if __name__ == "__main__":
    import tools.p2clab as C, tools.p2blab as B, tools.f1clab as F
    res, changed = {}, []
    for d in sorted(glob.glob(f"{P2}/*/")):
        piece = d.rstrip("/").split("/")[-1]; kd = json.load(open(d + "keydets.json"))
        fns = [j for j, bs in kd.items() if bs]; fb = [(cv2.imread(f"{d}raw_{int(j):04d}.jpg"), np.array(kd[j])) for j in fns]
        m, P0, P1 = run_set(piece, fb, fns, res, changed)
        if piece == C.PIECE:
            res[piece]["88 labelled, old"] = C.grade([x for p in P0 for x in p]); res[piece]["88 labelled, new"] = C.grade([x for p in P1 for x in p])
        if "reymersholm" in piece:
            for side in (0, 1):
                os.environ["IPANEMA_SIDE_LINES"] = str(side)
                res[piece][f"252 night key, side {side}"] = B.grade([m.predict_batch(B.KF[fn], np.array([B.KB[fn][j]]))[0] for (fn, j), _ in B.KEY])
            os.environ.pop("IPANEMA_SIDE_LINES", None)
        print(piece, json.dumps({k: v for k, v in res[piece].items() if k != "changes"}), flush=True)
    for g in ("p15u-vs-reymersholm-2026-09-18", "p15u-vs-spanga-2026-09-25", "SFKBP1109_s1200"):
        fr = F.load(f"{F.R}/{g}"); fb = [(f, np.array(b)) for _, f, b in fr]; fns = [fn for fn, _, _ in fr]
        run_set("kitprobe_" + g, fb, fns, res, changed); print(g, json.dumps({k: v for k, v in res["kitprobe_" + g].items() if k != "changes"}), flush=True)
    sheet(changed, f"{OUT}/changed_people.jpg")
    json.dump(res, open(f"{OUT}/p2dlab.json", "w"), indent=1, ensure_ascii=False)
