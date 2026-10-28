"""N0-28f: per-player average position / thirds / heat map with the last seen spot held (analytics.held_track_points,
IPANEMA_PLAYER_HOLD_S, off until M2) and the lab tools/n0_28f_playerpos.py."""
import os, sys, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from ipanema import analytics as AN


def _row(i, tm, x, y, gk=False):
    return [i, tm, np.array([x, y], float), None, None, gk]


def test_held_track_points_holds_then_drops():
    # track 1 seen frames 0-9 at x=80, then out of the picture; track 2 always seen (keeps frames non-empty)
    per = [[_row(2, "A", 30, 30)] + ([_row(1, "A", 80, 30)] if k < 10 else []) for k in range(40)]
    h = AN.held_track_points(per, 10.0, 1.0, L=105.0, W=68.0)
    assert len(h[2]) == 40
    assert len(h[1]) == 10 + 10 and np.allclose(h[1][-1], [80, 30])      # held 1 s = 10 frames after leaving
    assert len(AN.held_track_points(per, 10.0, 0.0)[1]) == 10              # hold 0 = seen spots only


def test_empty_frame_and_edge_clear():
    per = [[_row(1, "A", 80, 30), _row(2, "A", 104.5, 30)]] + [[_row(3, "B", 50, 30)]] * 5 + [[]] + [[_row(3, "B", 50, 30)]] * 5
    h = AN.held_track_points(per, 10.0, 10.0, L=105.0, W=68.0)
    assert len(h[1]) == 6          # held over the 5 frames, cleared by the empty frame
    assert len(h[2]) == 1          # on the goal line: never held


def _stats(per, fps=10.0):
    tracks = {}
    for k, rows in enumerate(per):
        for r in rows: tracks.setdefault(r[0], {})[k] = r[2]
    n = len(per); frames_ = [{"carrier": None, "team": None, "pressure_m": None} for _ in range(n)]
    import pandas as pd
    return AN.stats(per, frames_, [], pd.DataFrame(columns=["from", "to", "team", "completed", "kind", "progressive", "quality", "better_option"]),
                    tracks, [2] * n, fps, 105.0, 68.0, {"A": True, "B": False})


def _per_view():
    """player 1 really stands at x = 80 all the time but the camera shows him only while the ball is near (first 20 of
    every 100 frames); between, he is 'out of the picture' - and player 1 then comes back at x = 60 for 2 frames each
    cycle (the moments the ball is near him from deeper)"""
    per = []
    for k in range(400):
        rows = [_row(2, "A", 30, 30), _row(3, "B", 50, 40)]
        c = k % 100
        if c < 2: rows.append(_row(1, "A", 60, 30))
        elif c < 20: rows.append(_row(1, "A", 80, 30))
        per.append(rows)
    return per


def test_default_off_is_unchanged(monkeypatch):
    monkeypatch.setattr(AN, "PLAYER_HOLD_S", 0.0)
    st = _stats(_per_view()); p1 = [p for p in st["players"] if p["id"] == 1][0]
    assert p1["avg_x_m"] == round((2 * 60 + 18 * 80) / 20, 1)


def test_hold_moves_average_toward_last_seen(monkeypatch):
    monkeypatch.setattr(AN, "PLAYER_HOLD_S", 3.0)
    st = _stats(_per_view()); p1 = [p for p in st["players"] if p["id"] == 1][0]
    assert p1["avg_x_m"] > round((2 * 60 + 18 * 80) / 20, 1)              # held at 80 for 3 s after each sighting
    assert p1["pct_att_third"] > 90 and abs(sum(st["heatmaps"]["1"][i][j] for i in range(12) for j in range(8)) - 1) < 1e-3
    assert p1["time_visible_s"] == 8.0                                    # visible time still counts seen frames only


def test_env_switch_reads_seconds(monkeypatch):
    monkeypatch.setenv("IPANEMA_PLAYER_HOLD_S", "10")
    m = importlib.reload(AN); assert m.PLAYER_HOLD_S == 10.0
    monkeypatch.delenv("IPANEMA_PLAYER_HOLD_S"); m = importlib.reload(AN); assert m.PLAYER_HOLD_S == 0.0


def test_lab_dry_run_and_scores():
    import n0_28f_playerpos as LAB
    t = LAB.player_stats([[10, 10], [90, 10]]); o = LAB.player_stats([[90, 10]])
    s = LAB.score(o, t)
    assert abs(s["pos"] - 40) < 1e-6 and s["dx"] > 0 and abs(s["thirds"] - (50 + 0 + 50) / 3) < 1e-6
    r = LAB.dry()
    assert r["hold20"]["pos_timeweighted"] <= r["seen"]["pos_timeweighted"] and r["hold20"]["tv"] <= r["seen"]["tv"]
    assert r["seen"]["rank"] > 0.9 and r["piece"]["n"] > r["seen"]["n"]
