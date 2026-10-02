"""F1c (2 Oct): shadow-aware kit lightness (IPANEMA_KIT_LIGHT=local, and auto = local only in hard sun/shade, the new
default) vs the old reading (frame), offline, free. "people that change" = frame vs auto.
Labelled grounds (by eye, results/qa/kitprobe): Reymersholm night 252, Spånga 279; SFK-BP clip: how many change team.
AIK full match (results/qa/f1c/frames, from tools/f1c_frames.py): match model fitted on the 36 fit frames, graded on the
test frames against results/qa/f1c/aik_labels.json (by eye) when it exists; counts of A/B/neither per frame otherwise.
Pictures: people that change team, per ground; AIK team strips for both readings.
    python tools/f1clab.py   -> results/qa/f1c/f1clab.json + sheets"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
R = "results/qa/kitprobe"; F = "results/qa/f1c/frames"; OUT = "results/qa/f1c"; os.makedirs(OUT, exist_ok=True)
AIK = "p15u-vs-aik-2026-09-21-bd09"; Q = lambda *a: None

def load(d):
    B = json.load(open(f"{d}/boxes.json")); return [(fn, cv2.imread(f"{d}/{fn}"), np.array(B[fn]).reshape(-1, 4)) for fn in sorted(B)]

def grade(pred, labs, t1, t2):
    """team names are arbitrary: team t1 = whichever of A/B most of its players get"""
    pairs = list(zip(labs, pred))
    a = "A" if sum(l == t1 and p == "A" for l, p in pairs) >= sum(l == t1 and p == "B" for l, p in pairs) else "B"; b = "B" if a == "A" else "A"
    n1 = sum(l == t1 for l in labs); n2 = sum(l == t2 for l in labs); nO = sum(l == "O" for l in labs)
    return {"team right": sum((l == t1 and p == a) or (l == t2 and p == b) for l, p in pairs), "of": n1 + n2,
            "wrong team": sum((l == t1 and p == b) or (l == t2 and p == a) for l, p in pairs),
            "players called neither": sum(l in (t1, t2) and p not in ("A", "B") for l, p in pairs),
            "others kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{nO}"}

def sheet(crops, name, per_row=12):
    if not crops: return
    cs = [cv2.resize(c, (48, 96), interpolation=cv2.INTER_NEAREST) for c in crops]
    while len(cs) % per_row: cs.append(np.zeros_like(cs[0]))
    cv2.imwrite(name, np.vstack([np.hstack(cs[r:r + per_row]) for r in range(0, len(cs), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

def crop(f, b): return f[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])]

def fit_pred(fit_frames, test_frames, light):
    os.environ["IPANEMA_KIT_LIGHT"] = light
    m = K.KitTeamModel().fit_frames([(f, b) for _, f, b in fit_frames], log=Q)
    pred = {fn: m.predict_batch(f, b) if len(b) else [] for fn, f, b in test_frames}
    os.environ.pop("IPANEMA_KIT_LIGHT", None); return m, pred

def keyed(g, labfile, t1, t2, res):
    fr = load(f"{R}/{g}"); L = json.load(open(f"{R}/{labfile}"))
    if "items" not in L:                                                  # Reymersholm: two label files
        items = json.load(open(f"{R}/reym_label_items.json")); lab0 = json.load(open(f"{R}/reym_labels.json"))["labels_by_eye"]
        P7 = json.load(open(f"{R}/reym_labels_p7.json"))
        key = [(tuple(it), lab0[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7["items"], P7["labels"])]
    else: key = [(tuple(it), l) for it, l in zip(L["items"], L["labels"])]
    out = {}; preds = {}
    for light in ("frame", "local", "auto"):
        m, p = fit_pred(fr, fr, light); preds[light] = p; out.setdefault("light chosen by auto", m.light) if light == "auto" else None; out["shade spread"] = m.shade if light == "auto" else out.get("shade spread")
        out[light] = grade([p[fn][j] for (fn, j), _ in key], [l for _, l in key], t1, t2) if key else None
    res[g] = out; changed(g, fr, preds, res); print(g, json.dumps(out), flush=True)

def changed(g, fr, preds, res):
    ch = [(fn, f, b[j], preds["frame"][fn][j], preds["auto"][fn][j]) for fn, f, b in fr for j in range(len(b)) if preds["frame"][fn][j] != preds["auto"][fn][j]]
    res.setdefault(g, {})["people that change"] = len(ch); res[g]["people"] = sum(len(b) for _, _, b in fr)
    for a, z in (("A", "B"), ("B", "A")):
        sheet([crop(f, b) for _, f, b, x, y in ch if x == a and y == z][:96], f"{OUT}/{g}_changed_{a}to{z}.jpg")
    for nm, cond in (("into_neither", lambda x, y: y == "K"), ("out_of_neither", lambda x, y: x == "K")):
        sheet([crop(f, b) for _, f, b, x, y in ch if cond(x, y)][:96], f"{OUT}/{g}_{nm}.jpg")
    res[g]["change kinds"] = {f"{x}->{y}": sum(1 for *_, xx, yy in ch if (xx, yy) == (x, y)) for x, y in sorted({(c[3], c[4]) for c in ch}, key=str)}

def aik(res):
    if not os.path.exists(f"{F}/{AIK}/fit/boxes.json"): print("AIK frames not there yet"); return
    fit, test = load(f"{F}/{AIK}/fit"), load(f"{F}/{AIK}/test"); out = {}; preds = {}; models = {}
    lf = f"{OUT}/aik_labels.json"; L = json.load(open(lf)) if os.path.exists(lf) else None
    for light in ("frame", "local", "auto"):
        m, p = fit_pred(fit, test, light); preds[light] = p; models[light] = m
        if light == "auto": out["light chosen by auto"], out["shade spread"] = m.light, m.shade
        top = {fn: K.pitch_top(f) for fn, f, _ in test}
        on = [(fn, j) for fn, f, b in test for j in range(len(b)) if b[j][3] - b[j][1] >= 22 and K.feet_on_pitch(f, b[j], top[fn])]
        cnt = {t: round(sum(p[fn][j] == t for fn, j in on) / len(test), 2) for t in "ABK"}
        out[light] = {"per frame on the pitch (A dark, B light, K neither)": cnt, "groups": [int(x) for x in m.model["sizes"]]}
        if L: out[light]["key"] = grade([p[fn][j] for fn, j in map(tuple, L["items"])], L["labels"], "D", "W")
    res[AIK] = out; changed(AIK, test, preds, res)
    strips = [np.vstack([cv2.resize(models[l].strips[t], (576, 144)) for t in ("A", "B")]) for l in ("frame", "local")]
    for s, l in zip(strips, ("frame (old)", "local light")): cv2.rectangle(s, (0, 0), (200, 22), (0, 0, 0), -1); cv2.putText(s, l, (4, 16), 0, 0.5, (255, 255, 255), 1)
    cv2.imwrite(f"{OUT}/{AIK}_strips.jpg", np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(AIK, json.dumps(out), flush=True)

if __name__ == "__main__":
    res = {}
    only = os.environ.get("F1C_ONLY", "")
    if only in ("", "keys"):
        keyed("p15u-vs-reymersholm-2026-09-18", "reym_labels.json", "G", "W", res)
        keyed("p15u-vs-spanga-2026-09-25", "spanga_labels_p8.json", "D", "S", res)
        fr = load(f"{R}/SFKBP1109_s1200"); preds = {l: fit_pred(fr, fr, l)[1] for l in ("frame", "auto")}; changed("SFKBP1109_s1200", fr, preds, res)
        print("SFK", res["SFKBP1109_s1200"], flush=True)
    if only in ("", "aik"): aik(res)
    prev = json.load(open(f"{OUT}/f1clab.json")) if os.path.exists(f"{OUT}/f1clab.json") else {}
    prev.update(res); json.dump(prev, open(f"{OUT}/f1clab.json", "w"), indent=1)
