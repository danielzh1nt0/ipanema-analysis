"""P3 (5 Oct, local, $0): look vectors from the free-runner crops + pair sheets to grade joins by eye.
    python tools/p3_grade.py emb                 -> results/qa/p3/emb.json   (one colour vector per track)
    python tools/p3_grade.py sheets <lab.json>... -> results/qa/p3/pairs/<piece>_NN.jpg + pairs.json (every join proposed
                                                    by any of the given lab runs, numbered; grades go in grades.json)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2

CR = "results/qa/p3/crops"; OUT = "results/qa/p3"


def load_crops(name):
    J = json.load(open(f"{CR}/{name}.json")); S = cv2.imread(f"{CR}/{name}.jpg"); cw, ch, pr = J["cw"], J["ch"], J["per_row"]
    out = {}
    for c in J["cells"]:
        i = c["cell"]; im = S[(i // pr) * ch:(i // pr + 1) * ch, (i % pr) * cw:(i % pr + 1) * cw]
        out.setdefault(c["id"], []).append((c["k"], im))
    for v in out.values(): v.sort(key=lambda x: x[0])
    return out


def look(im):
    """shirt + shorts colour: HSV histograms of the middle of the box, grass and black padding left out."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); h, w = im.shape[:2]
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50)
    pad = im.max(axis=2) < 6
    vec = []
    for y0, y1 in ((0.14, 0.48), (0.48, 0.70)):
        sl = (slice(int(y0 * h), int(y1 * h)), slice(int(0.22 * w), int(0.78 * w)))
        m = ~(grass[sl] | pad[sl]); px = hsv[sl][m]
        if len(px) < 10: vec += [0.0] * (12 * 3 + 8); continue
        col = px[px[:, 1] > 45]
        hh = np.histogram2d(col[:, 0], col[:, 1], bins=[12, 3], range=[[0, 180], [45, 256]])[0].ravel() if len(col) else np.zeros(36)
        vv = np.histogram(px[:, 2], bins=8, range=(0, 256))[0]
        v = np.concatenate([hh, vv]).astype(float); vec += list(v / max(v.sum(), 1))
    v = np.sqrt(np.asarray(vec)); return v / max(np.linalg.norm(v), 1e-9)


def emb():
    E = {}
    for f in sorted(os.listdir(CR)):
        if not f.endswith(".json"): continue
        name = f[:-5]; C = load_crops(name)
        E[name] = {str(i): list(np.round(np.mean([look(im) for _, im in v], axis=0), 4)) for i, v in C.items()}
    json.dump(E, open(f"{OUT}/emb.json", "w")); print({k: len(v) for k, v in E.items()})


def sheets(labs):
    os.makedirs(f"{OUT}/pairs", exist_ok=True)
    L = [json.load(open(p)) for p in labs]; allp = {}
    for name in L[0]:
        prs = []
        for i, R in enumerate(L):
            for a, b, c, ad in R[name]["joins"]:
                if (a, b) not in prs: prs.append((a, b))
        if not prs: continue
        C = load_crops(name); cand = {(c[0], c[1]): c for c in L[0][name]["cands"]}
        panels = []
        for j, (a, b) in enumerate(prs):
            A = [im for _, im in C.get(a, [])[-2:]]; B = [im for _, im in C.get(b, [])[:2]]
            blank = np.zeros((144, 64, 3), np.uint8)
            A = (A + [blank, blank])[:2]; B = (B + [blank, blank])[:2]
            row = np.hstack(A + [np.full((144, 8, 3), 255, np.uint8)] + B)
            row = cv2.resize(row, (row.shape[1] * 2, 288), interpolation=cv2.INTER_NEAREST)
            lab = np.zeros((34, row.shape[1], 3), np.uint8); c = cand[(a, b)]
            cv2.putText(lab, f"#{j} {a}->{b} gap {c[2]}s {c[3]}m", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(np.vstack([lab, row]))
        allp[name] = [{"n": j, "a": a, "b": b, "gap": cand[(a, b)][2], "dist": cand[(a, b)][3]} for j, (a, b) in enumerate(prs)]
        for s in range(0, len(panels), 12):
            grp = panels[s:s + 12]; grp += [np.zeros_like(panels[0])] * (-len(grp) % 4)
            cv2.imwrite(f"{OUT}/pairs/{name}_{s // 12:02d}.jpg", np.vstack([np.hstack(grp[r:r + 4]) for r in range(0, len(grp), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(allp, open(f"{OUT}/pairs/pairs.json", "w"), indent=0); print({k: len(v) for k, v in allp.items()})


if __name__ == "__main__":
    if sys.argv[1] == "emb": emb()
    else: sheets(sys.argv[2:])
