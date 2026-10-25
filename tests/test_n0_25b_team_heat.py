"""N0-25b: team heat map in stats.teams[].heatmap (analytics.team_heatmaps, last seen spot held 20 s)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN


def _row(i, tm, x, y, gk=False):
    return [i, tm, np.array([x, y], float), None, None, gk]


def _per(n=200, view=None):
    """team A: 5 players at x = 10, 30, 50, 70, 90; the camera sees only x < view (if given)"""
    per = []
    for k in range(n):
        rows = [_row(i, "A", 10 + 20 * i, 30) for i in range(5) if view is None or 10 + 20 * i < view(k)]
        rows += [_row(100 + i, "B", 20 + 20 * i, 20) for i in range(4)] + [_row(99, "A", 2, 32, gk=True)]
        per.append(rows)
    return per


def test_grid_normalised_and_keeper_out():
    th = AN.team_heatmaps(_per(), 10.0, 105.0, 68.0, {"A": True, "B": False}, hold_s=0)
    g = np.array(th["A"]["grid"]); assert g.shape == (12, 8) and abs(g.sum() - 1) < 1e-3
    assert g[0].sum() < 1e-6                        # keeper at x = 2 is left out
    assert th["A"]["thirds_pct"] == [40, 20, 40] and th["B"]["thirds_pct"][::-1] == [25, 50, 25]


def test_hold_brings_back_players_out_of_view():
    """first half of the time the camera sees everything, then only x < 60: the seen map loses the far players, the held
    map keeps them for hold_s"""
    view = lambda k: 200 if k < 100 else 60
    seen = np.array(AN.team_heatmaps(_per(200, view), 10.0, 105.0, 68.0, hold_s=0)["A"]["grid"])
    held = np.array(AN.team_heatmaps(_per(200, view), 10.0, 105.0, 68.0, hold_s=20)["A"]["grid"])
    far = lambda g: g[int(65 / 105 * 12):].sum()
    assert far(held) > far(seen) + 0.1 and abs(far(held) - 0.4) < 0.02   # held 20 s = the whole second half here


def test_unseen_team_is_none_and_env_default():
    per = [[_row(1, "A", 50, 30), _row(2, "A", 52, 30), _row(3, "A", 54, 30)] for _ in range(20)]
    th = AN.team_heatmaps(per, 10.0, 105.0, 68.0)
    assert th["B"] is None and th["A"]["hold_s"] == AN.TEAM_HEAT_HOLD_S == float(os.environ.get("IPANEMA_TEAM_HEAT_HOLD_S", "20"))


def test_lab_dry_run():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import n0_25b_heat as H
    r = H.dry()
    for tm in "AB":
        assert r[tm]["hold5"]["tv"] <= r[tm]["frames"]["tv"] + 0.5


def test_moments_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); p = os.path.join(root, "results/qa/n0-25b/moments_key.json")
    if not os.path.exists(p): return
    key = json.load(open(p))
    for short in ("sfk", "aik"):
        M = json.load(open(os.path.join(root, f"results/qa/n0-25b/moments_{short}.json")))
        assert all(len(m["tiles"]) == 4 and m["id"] in key for m in M["moments"])


def test_in_stats_export(monkeypatch):
    """stats() writes teams[].heatmap (the app reads it instead of building one); hold 0 = only the seen players"""
    n = 60; per = _per(n); fps = 10.0
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    tracks = {}
    for k in range(n):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]
    st = AN.stats(per, frames_, [], [], tracks, np.full(n, 2), fps, 105.0, 68.0, {"A": True, "B": False})
    a = next(t for t in st["teams"] if t["team"] == "A")["heatmap"]
    assert np.array(a["grid"]).shape == (12, 8) and a["thirds_pct"] == [40, 20, 40] and a["hold_s"] == AN.TEAM_HEAT_HOLD_S
    json.dumps(st["teams"])                                                  # plain JSON, no numpy left
