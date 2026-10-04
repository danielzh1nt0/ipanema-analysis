"""4 Oct: Vallentuna (red vs black, hard shade) kit model offline: light mode frame / local / auto on this match's fit frames,
team counts on the test frames, team strips for the eye. Free, local.   PYTHONPATH=. python tools/vall_kits.py"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "tools"); from f1clab import load, fit_pred
from ipanema import kits as K
M = "p15u-vs-vallentuna-2026-10-03-6cce"; F = f"results/qa/f1c/frames/{M}"; OUT = "results/qa/f1c"
fit, test = load(f"{F}/fit"), load(f"{F}/test"); out = {}; models = {}
for light in ("frame", "local", "auto"):
    m, p = fit_pred(fit, test, light); models[light] = m
    top = {fn: K.pitch_top(f) for fn, f, _ in test}
    on = [(fn, j) for fn, f, b in test for j in range(len(b)) if b[j][3] - b[j][1] >= 22 and K.feet_on_pitch(f, b[j], top[fn])]
    cnt = {t: round(sum(p[fn][j] == t for fn, j in on) / len(test), 2) for t in "ABK"}
    out[light] = {"per frame on the pitch (A dark, B light, K neither)": cnt, "groups": [int(x) for x in m.model["sizes"]], "chosen": getattr(m, "light", light), "shade": getattr(m, "shade", None)}
    print(light, out[light], flush=True)
strips = [np.vstack([cv2.resize(models[l].strips[t], (576, 144)) for t in ("A", "B")]) for l in ("frame", "local")]
for s, l in zip(strips, ("frame (colour as is)", "local light")): cv2.rectangle(s, (0, 0), (220, 22), (0, 0, 0), -1); cv2.putText(s, l, (4, 16), 0, 0.5, (255, 255, 255), 1)
cv2.imwrite(f"{OUT}/{M}_strips.jpg", np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80]); json.dump(out, open(f"{OUT}/{M}_kits.json", "w"), indent=1)
