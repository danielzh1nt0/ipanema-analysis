"""K3c (5 Oct): every person in the 22 Vallentuna carrier frames, labelled by the kit model (hue mode + red-pixel share);
contact sheets per label so a person can count the mistakes by eye.  PYTHONPATH=. python tools/k3c_share_check.py <out>"""
import sys, os, json, cv2, numpy as np, collections
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import v3lab as L
from ipanema import kits as K
out = sys.argv[1] if len(sys.argv) > 1 else "results/qa/k3c"; os.makedirs(out, exist_ok=True)
m = L.fit_model(); print("hue mode", K.hue_mode(m.model), "cls", len(m.cls), "kit hue", K._kit_hue(m.model))
B = json.load(open("results/qa/v3/carriers/boxes.json")); by = collections.defaultdict(list); tot = collections.Counter()
for fn in sorted(B):
    f = cv2.imread(f"results/qa/v3/carriers/{fn}"); bx = [b[:4] for b in B[fn]["base"] if b[3] - b[1] >= 22]
    for b, lab in zip(bx, m.predict_batch(f, bx)):
        tot[lab] += 1; by[lab].append(cv2.resize(f[int(b[1]):int(b[3]), int(b[0]):int(b[2])], (50, 90)))
print(dict(tot))
for lab, cs in by.items():
    while len(cs) % 25: cs.append(np.full_like(cs[0], 255))
    cv2.imwrite(f"{out}/label_{lab}.jpg", np.vstack([np.hstack(cs[i:i + 25]) for i in range(0, len(cs), 25)]))
