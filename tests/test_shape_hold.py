"""N0-22e (22 Oct): team block length / width in stats.json (teams[].block_length_median_m, block_width_median_m) came only
from the players in the picture; our camera shows part of the pitch, so they read ~25% / ~20% short (Metrica pro tracking
seen through our clips' real camera views + our track breaks). Now a player who leaves the picture keeps his last seen
spot for analytics.SHAPE_HOLD_S (5) seconds - not a spot on the pitch's edge (people at the line, feet cut by the
picture). Line height keeps the seen players (already right). IPANEMA_SHAPE_HOLD_S=0 = old numbers."""
import os, sys, json, subprocess
import numpy as np, pytest

from ipanema import analytics as AN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 10.0
L, W = 105.0, 68.0


def _row(tid, team, x, y, gk=False):
    return [tid, team, np.array([x, y], float), np.zeros(2), None, gk]


def _team_frames(n, hide_from=None, hide_ids=(), xs=(20, 30, 40, 50, 60, 70), team="A", y=34.0, extra=None):
    """team A spread over xs (one player per x, ids 1..), a keeper at x 2; ids in hide_ids leave the picture at frame hide_from"""
    per = {}
    for k in range(n):
        rows = [_row(0, team, 2, 34, gk=True)]
        for i, x in enumerate(xs, start=1):
            if hide_from is not None and k >= hide_from and i in hide_ids: continue
            rows.append(_row(i, team, x, y + i))
        if extra: rows += extra(k)
        per[k] = rows
    return per


def test_held_for_hold_s_then_dropped():
    per = _team_frames(120, hide_from=20, hide_ids=(5, 6))                   # the two highest players leave the picture at 2 s
    held = AN.held_points(per, FPS, 5.0, L=L, W=W)
    assert len(held[10]["A"]) == 6 and np.ptp(held[10]["A"][:, 0]) == 50
    assert len(held[30]["A"]) == 6 and np.ptp(held[30]["A"][:, 0]) == 50     # 1 s later: still held at their last spots
    assert len(held[69]["A"]) == 6                                           # last seen at frame 19: 5.0 s later still in
    assert len(held[70]["A"]) == 4 and np.ptp(held[70]["A"][:, 0]) == 30     # 5.1 s: gone
    assert held[10]["B"] is None                                             # no B player seen: no reading
    assert all(not np.any(np.isclose(h[:, 0], 2)) for h in (held[10]["A"], held[70]["A"]))   # keeper left out


def test_spot_on_the_edge_not_held_and_empty_frame_clears():
    per = _team_frames(60, hide_from=20, hide_ids=(6,), extra=lambda k: [_row(9, "A", 90, 0.6)] if k < 20 else [])   # someone on the touchline
    held = AN.held_points(per, FPS, 5.0, L=L, W=W)
    assert any(np.allclose(p, (90, 0.6)) for p in held[10]["A"])            # while seen he counts (as before)
    assert not any(np.allclose(p, (90, 0.6)) for p in held[25]["A"])        # not held on the line
    assert any(np.allclose(p, (70, 40)) for p in held[25]["A"])             # the player who left mid-pitch is held
    held0 = AN.held_points(per, FPS, 5.0)                                    # no pitch size: no edge rule
    assert any(np.allclose(p, (90, 0.6)) for p in held0[25]["A"])
    per[30] = []                                                             # half-time / untrusted camera frame
    held = AN.held_points(per, FPS, 5.0, L=L, W=W)
    assert held[30]["A"] is None and len(held[31]["A"]) == 5                 # cleared: only the seen five after it


def test_needs_three_seen():
    per = {k: [_row(1, "A", 30, 30), _row(2, "A", 60, 30), _row(3, "A", 50, 20), _row(4, "A", 40, 50), _row(5, "A", 45, 10)] if k < 10
           else [_row(1, "A", 30, 30), _row(2, "A", 60, 30)] for k in range(20)}
    held = AN.held_points(per, FPS, 5.0, L=L, W=W)
    assert len(held[5]["A"]) == 5 and held[15]["A"] is None                  # two seen + three held is not a team reading


def _team_row(per, n, monkeypatch, hold):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", hold)
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    tracks = {}
    for k in range(n):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]
    st = AN.stats(per, frames_, [], [], tracks, np.full(n, 2), FPS, L, W, {"A": True, "B": False})
    return next(t for t in st["teams"] if t["team"] == "A")


def test_stats_block_length_with_held_players(monkeypatch):
    # the camera follows play: for 2 s of every 3 the two deepest players are out of the picture
    n = 300
    per = {}
    for k in range(n):
        out = (k % 30) >= 10
        per[k] = [_row(0, "A", 2, 34, gk=True)] + [_row(i, "A", x, 30 + i) for i, x in enumerate((20, 30, 40, 50, 60, 70, 75), start=1) if not (out and i in (1, 2))]
    new = _team_row(per, n, monkeypatch, 5.0); old = _team_row(per, n, monkeypatch, 0.0)
    assert old["block_length_median_m"] == pytest.approx(35.0)                 # seen only: 40..75 most of the time
    assert new["block_length_median_m"] == pytest.approx(55.0)                 # the whole team: 20..75
    assert new["def_line_height_median_m"] == old["def_line_height_median_m"]   # line height: seen players, unchanged
    assert new["block_hold_s"] == 5.0 and old["block_hold_s"] == 0.0


def test_old_switch_same_as_before(monkeypatch):
    rng = np.random.default_rng(0); n = 200
    per = {k: [_row(i, "A" if i < 8 else "B", *rng.uniform((5, 5), (100, 63))) for i in range(16) if rng.random() < 0.7] for k in range(n)}
    old = _team_row(per, n, monkeypatch, 0.0)
    lens = []
    for k in range(n):
        p = np.array([r[2] for r in per[k] if r[1] == "A" and not r[5]])
        if len(p) >= 5: lens.append(np.ptp(p[:, 0]))
    assert old["block_length_median_m"] == round(float(np.median(lens)), 2)


def test_metrica_tool_dry_run():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_22e_shape_view.py"), "--dry"], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["home"]["seen5"]["length"] < d["home"]["all"]["length"]          # a half view shortens the seen team
