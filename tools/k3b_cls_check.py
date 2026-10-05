"""K3b (5 Oct): on the 22 Vallentuna carrier frames, every person labelled with the per-player classifier (left dot) and by colour only (right dot); red = A (SFK, black), blue = B (red kit), grey = neither.
    PYTHONPATH=. python tools/k3b_cls_check.py <out_dir>"""
import os, sys, json, cv2, numpy as np, collections
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import v3lab as L
os.environ["IPANEMA_KIT_HUEMODE"]="1"
from ipanema import kits as K
_h = K.hue_mode; K.hue_mode = lambda m: False   # fit WITH the classifier to compare
m = L.fit_model(); K.hue_mode = _h; cls = m.cls; print("cls runs", len(cls), "swap", m.swap, "teams", [np.round(t,1).tolist() for t in m.model["teams"]])
D = "results/qa/v3/carriers"; B = json.load(open(f"{D}/boxes.json")); S = sys.argv[1]
tot = collections.Counter(); per = collections.Counter()
for fn in sorted(B):
    f = cv2.imread(f"{D}/{fn}"); bx = [b for b in B[fn]["base"]]
    out = {}
    for name, c in (("cls", cls), ("colour", [])):
        m.cls = c; out[name] = m.predict_batch(f, [list(b[:4]) for b in bx])
    m.cls = cls
    g = f.copy()
    for b, a, c in zip(bx, out["cls"], out["colour"]):
        x, y = int((b[0]+b[2])/2), int(b[3])
        col = {"A": (0,0,255), "B": (255,120,0)}.get(a, (160,160,160)); cv2.circle(g, (x-6, y+6), 7, col, -1)
        col = {"A": (0,0,255), "B": (255,120,0)}.get(c, (160,160,160)); cv2.circle(g, (x+8, y+6), 7, col, -1)
        tot[(a, c)] += 1; per[fn] += (a=="B" and c=="A")
    cv2.imwrite(f"{S}/{fn}", cv2.resize(g, (1280, 720)))
print(tot); print(per.most_common(5))
