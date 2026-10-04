"""P2e (4 Oct): on a ground without calibration (tracktest), positions are screen positions, so the goalmouth keeper rules
must not fire: a 'K' track (referee) at the picture's left edge stays 'K' and is removed, and a player of team B at the
left edge keeps his own team instead of becoming the left team's keeper. With calibration (default) nothing changes."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR

L, W, fps = 106.0, 64.0, 30.0

def make():
    per = {k: [] for k in range(120)}
    for k in range(120):
        for i in range(4): per[k].append([i, "A", np.array([30.0 + i, 10.0 + 5 * i]), np.zeros(2), np.array([0, 0, 10, 30.]), False])
        for i in range(4): per[k].append([10 + i, "B", np.array([70.0 + i, 10.0 + 5 * i]), np.zeros(2), np.array([0, 0, 10, 30.]), False])
        per[k].append([99, "K", np.array([1.0 + 6.0 * k / 120, 35.0]), np.zeros(2), np.array([0, 0, 10, 30.]), False])    # referee at the left screen edge
        per[k].append([50, "B", np.array([4.0, 30.0 + 8.0 * k / 120]), np.zeros(2), np.array([0, 0, 10, 30.]), False])    # white player at the left edge
    return per

def test_zone_keepers_default_unchanged():
    per, info = TR.clean(make(), L, W, fps, log=lambda *a: None)
    ids = {r[0]: r for r in per[60]}
    assert ids[50][1] == "B" and not ids[50][5]                                     # 4 Oct: a B player in A's goalmouth keeps his team (was repainted as A's keeper)
    assert 99 in ids and ids[99][1] == "A"                                         # old behaviour: referee in the 'box' made a keeper

def test_screen_positions_no_zone_keepers():
    per, info = TR.clean(make(), L, W, fps, log=lambda *a: None, keepers_by_zone=False)
    ids = {r[0]: r for r in per[60]}
    assert info["keepers"] == []
    assert ids[50][1] == "B" and not ids[50][5], "edge player keeps his own team"
    assert 99 not in ids, "referee removed as a non-team kit"

if __name__ == "__main__":
    test_zone_keepers_default_unchanged(); test_screen_positions_no_zone_keepers(); print("OK")
