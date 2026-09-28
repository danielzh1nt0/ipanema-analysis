"""P6 (29 Sep): does the per-track team vote disagree with each frame's own colour reading? Reymersholm Kaggle rows.
    python tools/p6lab.py"""
import json, gzip, collections, numpy as np
R = json.load(gzip.open("results/kaggle/track_reym/rows_all.json.gz", "rt"))
rows = R["rows"]["new, RF-DETR"]; raw = R["raw_team"]["new, RF-DETR"]
agree = collections.Counter(); per_track = collections.defaultdict(collections.Counter)
for k, rs in rows.items():
    for (pid, team, px, fl), rt in zip(rs, raw.get(k, [])):
        if rt is None or fl: continue
        agree[(rt, team)] += 1; per_track[pid][rt] += 1
tot = sum(agree.values()); print("frame reading -> final label:", {f"{a}->{b}": v for (a, b), v in sorted(agree.items())}, "of", tot)
print("readings where final team differs from this frame's A/B reading:", sum(v for (a, b), v in agree.items() if a in "AB" and b in "AB" and a != b) / max(1, tot))
mixed = [c for c in per_track.values() if c["A"] >= 3 and c["B"] >= 3]
print(f"tracks: {len(per_track)}; tracks with both A and B read >= 3 times (likely identity swaps or unsure colour): {len(mixed)}")
tA = sum(1 for c in per_track.values() if c["A"] > c["B"]); tB = sum(1 for c in per_track.values() if c["B"] > c["A"]); print("tracks mostly A / mostly B:", tA, tB)
rd = collections.Counter(a for (a, b) in agree.elements()); print("all frame readings:", dict(rd))
