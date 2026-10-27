"""N0-27a (27 Oct): 'a better pass was open' (passes[].better_option, teams[] / players[] better_option_count).
The valuation is pulled out of analytics.passes into analytics.best_option unchanged (so the Metrica check runs the very
same rule); IPANEMA_BETTER=withhold writes None for both counts (the app shows Withheld)."""
import os, sys, numpy as np
from ipanema import analytics as AN

FPS = 25.0
TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")


def _rows(spots):
    return [[t, "A" if t < 200 else "B", np.array(xy, float), None, None, False] for t, xy in spots.items()]


SPOTS = {101: (30.0, 30.0), 102: (35.0, 30.0), 103: (55.0, 40.0), 201: (45.0, 20.0), 202: (60.0, 10.0), 203: (100.0, 34.0)}


def test_best_option_prefers_free_forward_mate():
    rows = _rows(SPOTS); ln = AN.lanes([rows], [{"carrier": 101}], {"A": True, "B": False})[0]
    bo = AN.best_option(rows, 101, "A", 1, ln, 102)
    assert bo["best_to"] == 103 and bo["better_option"] and bo["played_value"] == 10.0 and bo["best_value"] == 34.0
    assert AN.best_option(rows, 101, "A", 1, ln, 103)["better_option"] is False      # played the best one
    assert AN.best_option(rows, 999, "A", 1, ln, 102) == {}                            # carrier not in the picture


def test_blocked_lane_costs_25():
    sp = dict(SPOTS); sp[201] = (45.0, 35.0)                                            # an opponent on the line to 103
    rows = _rows(sp); ln = AN.lanes([rows], [{"carrier": 101}], {"A": True, "B": False})[0]
    v = AN.option_values(rows, 101, "A", 1, ln)
    assert v[103]["open"] is False and v[102]["open"] is True
    assert AN.best_option(rows, 101, "A", 1, ln, 102)["better_option"] is False        # 34 - 25 = 9 < 10 + 6


def _pass_setup():
    n = 60; per = {k: _rows(SPOTS) for k in range(n)}
    frames_ = [{"carrier": 101 if k < 25 else (102 if k >= 35 else None), "team": "A" if (k < 25 or k >= 35) else None, "pressure_m": None, "near_opps": 0, "pos": None} for k in range(n)]
    return per, frames_


def test_passes_uses_best_option():
    per, frames_ = _pass_setup(); ar = {"A": True, "B": False}
    ln = AN.lanes(per, frames_, ar)
    ps, tracks = AN.passes(per, frames_, [], ln, ar, FPS, min_pass_m=0)
    assert len(ps) == 1 and ps[0]["from"] == 101 and ps[0]["to"] == 102
    assert ps[0]["better_option"] and ps[0]["best_to"] == 103
    want = AN.best_option(per[24], 101, "A", 1, ln[24], 102)
    assert all(ps[0][k] == v for k, v in want.items())


def test_withhold_counts(monkeypatch):
    per, frames_ = _pass_setup(); ar = {"A": True, "B": False}
    ps, tracks = AN.passes(per, frames_, [], AN.lanes(per, frames_, ar), ar, FPS, min_pass_m=0)
    state = np.array([0] * len(frames_))
    on = AN.stats(per, frames_, [], ps, tracks, state, FPS, 105.0, 68.0, ar)
    assert next(t for t in on["teams"] if t["team"] == "A")["better_option_count"] == 1
    assert next(p for p in on["players"] if p["id"] == 101)["better_option_count"] == 1
    monkeypatch.setattr(AN, "BETTER", "withhold")
    off = AN.stats(per, frames_, [], ps, tracks, state, FPS, 105.0, 68.0, ar)
    assert all(t["better_option_count"] is None for t in off["teams"]) and all(p["better_option_count"] is None for p in off["players"])
    assert [t["passes"] for t in off["teams"]] == [t["passes"] for t in on["teams"]]


def test_lab_dry():
    sys.path.insert(0, TOOLS)
    import n0_27a_better_option as B
    assert B.dry()[0]["better_option"]


def test_rule_variant_app_equals_lanes():
    sys.path.insert(0, TOOLS)
    import n0_27a_rule as R
    rows = _rows(SPOTS); ln = AN.lanes([rows], [{"carrier": 101}], {"A": True, "B": False})[0]
    assert R.flag(rows, 101, "A", 1, 102) == AN.best_option(rows, 101, "A", 1, ln, 102)["better_option"]
    assert R.flag(rows, 101, "A", 1, 102, reach=10.0) is False                          # 103 is 26.9 m away
