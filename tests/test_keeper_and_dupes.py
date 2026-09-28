"""28 Sep: (1) a box inside a bigger box is dropped; (2) a 'K' track living in a penalty area becomes that end's keeper;
a 'K' track in midfield (referee) is removed"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR
keep = TR.drop_contained([[100, 100, 140, 220], [102, 105, 138, 160], [300, 100, 340, 220], [335, 100, 375, 220]])
assert keep.tolist() == [True, False, True, True], keep
L, W, fps = 106.0, 64.0, 30.0
per = {k: [] for k in range(120)}
for k in range(120):
    for i in range(5): per[k].append([i, "A", np.array([30.0 + i, 10.0 + 5 * i]), np.zeros(2), np.array([0, 0, 10, 30.]), False])
    for i in range(5): per[k].append([10 + i, "B", np.array([70.0 + i, 10.0 + 5 * i]), np.zeros(2), np.array([0, 0, 10, 30.]), False])
    per[k].append([99, "K", np.array([10.0, 32.0]), np.zeros(2), np.array([0, 0, 10, 30.]), False])   # keeper in the left box
    per[k].append([98, "K", np.array([53.0, 30.0]), np.zeros(2), np.array([0, 0, 10, 30.]), False])   # referee in midfield
per, info = TR.clean(per, L, W, fps, log=lambda *a: None)
ids = {r[0]: r for r in per[60]}
assert 99 in ids and ids[99][1] == info["left_team"] and ids[99][5], "keeper kept, on the team defending left"
assert 98 not in ids, "referee removed"
print("OK")
