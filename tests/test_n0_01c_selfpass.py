"""N0-01c (1 Nov): a pass from a player to himself makes no link in the pass network; the Metrica lab's matching and the
blind film-strip tools work."""
import os, sys, importlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_01c_selfpass as S
import n0_01c_moments as M
import n0_01c_frames as F
from ipanema import metrics as MT


def _ps():
    return [{"team": "A", "from": 1, "to": 2, "completed": True, "kind": "forward", "progressive": False, "from_m": [10, 10], "to_m": [20, 10]},
            {"team": "A", "from": 2, "to": 2, "completed": True, "kind": "sideways", "progressive": False, "from_m": [20, 10], "to_m": [22, 16]},
            {"team": "A", "from": 2, "to": 1, "completed": True, "kind": "backward", "progressive": False, "from_m": [22, 16], "to_m": [12, 12]}]


def _pl():
    return [{"id": 1, "avg_x_m": 11, "avg_y_m": 11}, {"id": 2, "avg_x_m": 21, "avg_y_m": 12}]


def test_network_skips_self_loops():
    net = MT.pass_network(_ps(), _pl())
    assert all(e["from"] != e["to"] for e in net["A"]["edges"])
    assert sum(e["count"] for e in net["A"]["edges"]) == 2
    # the node of player 2 no longer averages in the self pass's spots
    assert net["A"]["nodes"]["2"]["n"] == 2


def test_network_old_rule_keeps_them(monkeypatch):
    monkeypatch.setattr(MT, "NET_SELF", True)
    net = MT.pass_network(_ps(), _pl())
    assert any(e["from"] == e["to"] == 2 for e in net["A"]["edges"])


def test_metrica_matching_dry():
    t = S.dry()
    assert t["self"]["n"] == 2 and t["self"]["real"] == 1 and t["self"]["same_passer"] == 1
    assert t["other"]["real"] == 2


def test_box_holds_both_points():
    b = M.box([100, 900], [700, 950])
    assert b[0] <= 100 and b[0] + b[2] >= 700 and abs(b[2] / b[3] - 16 / 9) < 0.02
    assert b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= 1920 and b[1] + b[3] <= 1080


def test_centres_follow_ball_and_stay_in_frame():
    fr = [{"t": k * 0.1, "ball": {"px": [100 + 100 * k, 50]}} for k in range(30)]
    ts = [f["t"] for f in fr]
    tl = M.centres(fr, ts, 0.0, 2.5)
    assert len(tl) == 6 and tl[0] == [0, 0] and tl[-1][0] > tl[0][0]
    assert all(0 <= x <= 1920 - 720 and 0 <= y <= 1080 - 405 for x, y in tl)


def test_sheet_shape():
    import cv2 as cv
    m = {"id": "s00", "t0": 0.0, "t1": 1.0, "passer_px": [500, 500], "box": [0, 0, 960, 540], "tl": [[100, 100]] * 6}
    frs = [(t, np.zeros((1080, 1920, 3), np.uint8)) for t in F.times(m)]
    assert F.sheet(cv, np, frs, m).shape == (810, 2160, 3)


def _passes_with(monkeypatch, keep_self):
    from ipanema import analytics as AN
    eps = [(0, 10, 1, "A"), (20, 30, 1, "A"), (40, 50, 2, "A")]           # 1 touches, 1 again 8 m on, then 2
    monkeypatch.setattr(AN, "touch_episodes", lambda *a, **k: eps)
    monkeypatch.setattr(AN, "PASS_SELF", keep_self)
    per = {k: [[1, "A", np.array([10.0 + (8 if k >= 20 else 0), 30.0])], [2, "A", np.array([30.0, 30.0])]] for k in range(60)}
    out, _ = AN.passes(per, [None] * 60, [], {}, {"A": True, "B": False}, 25.0)
    return [(p["from"], p["to"]) for p in out]


def test_passes_drop_self(monkeypatch):
    assert _passes_with(monkeypatch, False) == [(1, 2)]
    assert _passes_with(monkeypatch, True) == [(1, 1), (1, 2)]
