"""Veo import: team from the attacking end, goals checked (a kick-off-minute goal with no restart after it is rejected)."""
import numpy as np, tempfile, os
from ipanema import veo as VEO
from ipanema.metrics import extra_events

def test_veo_build():
    fps, L, W = 10.0, 120.0, 70.0; attack_right = {"A": True, "B": False}
    ballm = {}; per = {}
    for t, x in ((600, 100.0), (2619, 105.0), (4422, 12.0), (28, 30.0)):           # action at A's attacking end (x>60) or B's (x<60)
        for k in range(int((t - 1) * fps), int((t + 1) * fps)): ballm[k] = np.array([x, 35.0])
    for k in range(int(5000 * fps), int(5020 * fps)): per[k] = [[i, "A", np.array([20.0 + i, 30.0]), None, None, False] for i in range(6)]   # no ball, players at B's end
    restarts = [{"t": 2700.0, "x_m": 60.0, "y_m": 35.0}, {"t": 4500.0, "x_m": 61.0, "y_m": 34.0}]      # kick-offs after the real goals
    p = os.path.join(tempfile.mkdtemp(), "h.txt"); open(p, "w").write("# test\n28 goal\n28 shot\n600 shot\n2618 shot\n2619 goal\n4421 shot\n4422 goal\n5021 shot\n")
    shots, rej = VEO.build(VEO.load(p), fps, ballm, per, L, W, attack_right, restarts, periods=[{"t_start": 0.0}, {"t_start": 3509.0}], log=lambda *a: None)
    assert [g["t"] for g in rej] == [28]                                               # the kick-off-minute goal tag is rejected
    by_t = {s["t"]: s for s in shots}
    assert by_t[28.0]["goal"] is False and by_t[28.0]["team"] == "B"                     # ...but the shot at 0:28 stays
    assert by_t[2619.0]["goal"] and by_t[2619.0]["team"] == "A" and 2618.0 not in by_t   # goal replaces its paired shot
    assert by_t[4422.0]["goal"] and by_t[4422.0]["team"] == "B"
    assert by_t[5021.0]["team"] == "B" and by_t[5021.0]["located_by"] == "players" and by_t[5021.0]["distance_m"] is None
    ev = extra_events({"shots": shots, "high_turnovers": []})
    assert any(e["type"] == "goal" and "from Veo" in e["subtitle"] for e in ev)
    assert VEO.score_detector([{"t": 601.0}, {"t": 900.0}], shots) == {"veo_shots": len(shots), "found": 1, "ours": 2, "real": 1}
