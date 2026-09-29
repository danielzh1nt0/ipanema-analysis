"""P7 (29 Sep): off-pitch people on grounds without calibration. Compares the old grass-under-the-feet test (on_grass) with
the new pitch-edge test (pitch_top / feet_on_pitch) on the 24 saved frames per ground (results/qa/kitprobe/<ground>/):
how many people each keeps, kit accuracy on the 71 Reymersholm players labelled by eye, the off-pitch "team" share, and
picture sheets (green = counted, red = dropped, magenta = pitch edge). Free, local."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K
R = "results/qa/kitprobe"; OUT = "results/qa/p7"; os.makedirs(OUT, exist_ok=True)
GROUNDS = ["p15u-vs-reymersholm-2026-09-18", "p15u-vs-spanga-2026-09-25", "SFKBP1109_s1200"]
def load(g):
    B = json.load(open(f"{R}/{g}/boxes.json")); return {fn: cv2.imread(f"{R}/{g}/{fn}") for fn in sorted(B)}, B
items = [tuple(x) for x in json.load(open(f"{R}/reym_label_items.json"))]
labels = json.load(open(f"{R}/reym_labels.json"))["labels_by_eye"]
P7L = json.load(open(f"{R}/reym_labels_p7.json"))                              # 222 more on-pitch people, by eye (29 Sep)
ALL = [(tuple(it), labels[str(i)]) for i, it in enumerate(items)] + [(tuple(it), l) for it, l in zip(P7L["items"], P7L["labels"])]
def grade(m, fr, B, pairs_src):
    pred = [m.predict_batch(fr[fn], [B[fn][j]])[0] for (fn, j), _ in pairs_src]; pairs = [(l, p) for (_, l), p in zip(pairs_src, pred)]
    green = "A" if sum(l == "G" and p == "A" for l, p in pairs) >= sum(l == "G" and p == "B" for l, p in pairs) else "B"; white = "B" if green == "A" else "A"
    nG = sum(l == "G" for l, _ in pairs); nW = sum(l == "W" for l, _ in pairs); nO = sum(l == "O" for l, _ in pairs)
    return {"team right": f"{sum((l == 'G' and p == green) or (l == 'W' and p == white) for l, p in pairs)}/{nG + nW}",
            "whites read green": f"{sum(l == 'W' and p == green for l, p in pairs)}/{nW}", "greens read white": f"{sum(l == 'G' and p == white for l, p in pairs)}/{nG}",
            "players read 'neither'": sum(l in 'GW' and p not in ('A', 'B') for l, p in pairs), "others kept out": f"{sum(l == 'O' and p not in ('A', 'B') for l, p in pairs)}/{nO}"}
res = {}
for test in ("grass", "edge"):
  for gk in (False, True):
    for g in GROUNDS:
        fr, B = load(g); fns = sorted(B)
        kept = sum(K.feet_on_pitch(fr[fn], b, K.pitch_top(fr[fn])) if test == "edge" else K.on_grass(fr[fn], b) for fn in fns for b in B[fn])
        m = K.KitTeamModel().fit_frames([(fr[fn], np.array(B[fn])) for fn in fns], log=lambda *a: None, pitch_test=test, green_kit=gk)
        m.offpitch = True
        allp = [x for fn in fns for x in m.predict_batch(fr[fn], np.array(B[fn]))]
        r = {"people": sum(len(B[fn]) for fn in fns), "on pitch (kept)": int(kept), "A/B/K/O": [allp.count(t) for t in "ABKO"]}
        if g.startswith("p15u-vs-reym"):
            m.offpitch = False                                                  # grade colour only
            r["old 93 (P5 set)"] = grade(m, fr, B, ALL[:len(items)]); r["all 315 labelled"] = grade(m, fr, B, ALL)
        cv2.imwrite(f"{OUT}/{g}_{test}{'_greenkit' if gk else ''}_kits.jpg", np.vstack([cv2.resize(m.strips[t], (576, 144)) for t in ("A", "B")]))
        res[f"{test}{' green_kit' if gk else ''} | {g}"] = r; print(test, gk, g, json.dumps(r), flush=True)
json.dump(res, open(f"{OUT}/p7lab.json", "w"), indent=1)
for g in GROUNDS:                                                              # picture sheets of the new test, 4 frames per sheet
    fr, B = load(g); fns = sorted(B); tiles = []
    for fn in fns:
        f = fr[fn].copy(); top = K.pitch_top(f)
        cv2.polylines(f, [np.c_[np.arange(0, f.shape[1], 8), top[::8]].astype(np.int32)], False, (255, 0, 255), 3)
        for j, b in enumerate(B[fn]):
            x1, y1, x2, y2 = map(int, b); cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0) if K.feet_on_pitch(f, b, top) else (0, 0, 255), 3)
        cv2.putText(f, fn, (20, 60), 0, 2, (255, 255, 255), 5); tiles.append(cv2.resize(f, (960, 540)))
    for i in range(0, len(tiles), 4):
        t = tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))
        cv2.imwrite(f"{OUT}/{g}_edge_{i // 4}.jpg", np.vstack([np.hstack(t[:2]), np.hstack(t[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 70])
