"""28 Sep: tracking.fill_gaps - fills short gaps inside a track, never long ones, never on top of a player already there"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR
fps = 30.0
def row(tid, team, m, px): return [tid, team, np.array(m, float), np.array(px, float), None, False]
per = {k: [] for k in range(200)}
for k in list(range(0, 20)) + list(range(30, 60)): per[k].append(row(1, "A", [10 + 0.1 * k, 20], [100 + k, 500]))      # gap of 10 frames
for k in list(range(0, 10)) + list(range(100, 110)): per[k].append(row(2, "B", [50, 30], [900, 400]))                   # gap of 90 frames: too long
for k in range(0, 60):
    if not (40 <= k < 50): per[k].append(row(3, "B", [70, 10], [1500, 300]))
for k in range(40, 50): per[k].append(row(9, "B", [70.4, 10.2], [1502, 302]))                                          # same person, new id, during the gap
H = {k: np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1]]) for k in range(200)}
per, n = TR.fill_gaps(per, fps, 1.0, H=H)
f1 = [r for k in range(20, 30) for r in per[k] if r[0] == 1]
assert len(f1) == 10 and all(len(r) == 7 and r[6] == "filled" for r in f1), "10-frame gap must be filled and marked"
assert abs(f1[0][2][0] - (10 + 0.1 * 20)) < 1e-6 and abs(f1[-1][2][0] - (10 + 0.1 * 29)) < 1e-6, "straight line in metres"
assert np.allclose(f1[0][3], H[20][:2, :2] @ f1[0][2]), "pixels through the frame's camera"
assert not any(r[0] == 2 for k in range(10, 100) for r in per[k]), "90-frame gap must not be filled"
assert not any(r[0] == 3 for k in range(40, 50) for r in per[k]), "no fill on top of the same player under a new id"
assert n == 10, n
print("OK")
# an id that jumps 20 m in 10 frames is two people: never filled
per2 = {k: [] for k in range(40)}
for k in range(0, 10): per2[k].append(row(5, "A", [10, 10], [100, 100]))
for k in range(20, 30): per2[k].append(row(5, "A", [30, 10], [300, 100]))
per2, n2 = TR.fill_gaps(per2, fps, 1.0)
assert n2 == 0, n2
print("OK speed guard")
