import os, sys, json, cv2, numpy as np
sys.path.insert(0, "/home/claude/ipanema-analysis"); os.chdir("/home/claude/ipanema-analysis")
from ipanema import kits as K
S = "/tmp/claude-0/-home-claude-ipanema-analysis/a36bb58e-f5d2-59cb-921d-5e633c5545db/scratchpad"
g = "p15u-vs-reymersholm-2026-09-18"; D = f"results/qa/kitprobe/{g}"; B = json.load(open(f"{D}/boxes.json")); fns = sorted(B)
orig = {fn: cv2.imread(f"{D}/{fn}") for fn in fns}
# round-trip through the same writer tracktest uses (mp4v), all 24 frames as one short video
w = cv2.VideoWriter(f"{S}/rt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 29.97, (1920, 1080))
for fn in fns:
    for _ in range(3): w.write(orig[fn])                                     # a few copies so the encoder settles
w.release(); cap = cv2.VideoCapture(f"{S}/rt.mp4"); rt = {}
for fn in fns:
    fr = [cap.read()[1] for _ in range(3)]; rt[fn] = fr[1]
items = [tuple(x) for x in json.load(open("results/qa/kitprobe/reym_label_items.json"))]
labels = json.load(open("results/qa/kitprobe/reym_labels.json"))["labels_by_eye"]
for name, fr in (("original frames", orig), ("after mp4v re-save", rt)):
    m = K.KitTeamModel().fit_frames([(fr[fn], np.array(B[fn])) for fn in fns], log=lambda *a: None)
    pred = [m.predict_batch(fr[fn], [B[fn][j]])[0] for fn, j in items]; pairs = [(labels[str(i)], p) for i, p in enumerate(pred)]
    green = "A" if sum(1 for l, p in pairs if l == "G" and p == "A") >= sum(1 for l, p in pairs if l == "G" and p == "B") else "B"; white = "B" if green == "A" else "A"
    ok = sum(1 for l, p in pairs if (l == "G" and p == green) or (l == "W" and p == white)); wg = sum(1 for l, p in pairs if l == "W" and p == green)
    allp = [m.predict_batch(fr[fn], np.array(B[fn])) for fn in fns]; flat = [x for a in allp for x in a]
    print(f"{name}: team right {ok}/71, whites called green {wg}/31; all people A/B/K = {flat.count('A')}/{flat.count('B')}/{sum(1 for x in flat if x not in ('A','B'))}")
print("file size MB", round(os.path.getsize(f'{S}/rt.mp4') / 1e6, 1), "for", 3 * len(fns), "frames")
