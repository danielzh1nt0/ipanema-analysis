"""28 Sep: kits.py on the 80 all-footage pictures: per match, crops grouped by the label they get -> one sheet per match"""
import sys, os, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
R = "results/qa/players"; OUT = sys.argv[1] if len(sys.argv) > 1 else "results/qa/kits"; os.makedirs(OUT, exist_ok=True); rep = {}
for m in sorted(d for d in os.listdir(R) if os.path.isdir(f"{R}/{d}") and d != "players_run2" and os.path.exists(f"{R}/{d}/boxes.json") and any(n.startswith("raw_") for n in os.listdir(f"{R}/{d}"))):
    it = json.load(open(f"{R}/{m}/boxes.json")); people = []
    for x in it:
        im = cv2.imread(f"{R}/{m}/raw_f{x['frame']:07d}.jpg"); s = im.shape[1] / x["size"][0]; gl = K.grass_lab(im)   # unmarked frame
        for b in x["boxes"]:
            if b[5] == "at 0.10": continue
            bb = [v * s for v in b[:4]]
            if bb[3] - bb[1] < 22: continue                                    # too small to read a shirt
            people.append((im, bb, K.torso_feature(im, bb, gl)))
    model = K.fit([p[2] for p in people]); groups = {"A": [], "B": [], "other": [], None: []}
    for im, bb, f in people:
        x1, y1, x2, y2 = [int(v) for v in bb]; c = im[max(0, y1 + 2):max(0, y2 - 2), max(0, x1 + 2):max(0, x2 - 2)]
        if c.size: groups[K.classify(model, f)].append(cv2.resize(c, (32, 72), interpolation=cv2.INTER_NEAREST))
    rows = []
    for g, col in (("A", (0, 0, 255)), ("B", (255, 128, 0)), ("other", (255, 0, 255))):
        t = groups[g][:30] + [np.zeros((72, 32, 3), np.uint8)] * max(0, 30 - len(groups[g]))
        row = np.hstack(t); lab = np.full((72, 110, 3), 30, np.uint8); cv2.putText(lab, f"{g} {len(groups[g])}", (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        rows.append(np.hstack([lab, row]))
    cv2.imwrite(f"{OUT}/{m}.jpg", np.vstack(rows)); rep[m] = {g if g else "unreadable": len(v) for g, v in groups.items()}; print(m[:30], rep[m], "sizes", model["sizes"])
json.dump(rep, open(f"{OUT}/summary.json", "w"), indent=1)
