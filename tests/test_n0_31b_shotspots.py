import os, json, importlib.util
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("n0_31b", os.path.join(REPO, "tools/n0_31b_shotspots.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def _st(shots, ar={"A": True, "B": False}):
    return {"pitch": {"length": 106.0, "width": 64.0}, "periods": [{"attack_right": ar}], "metrics": {"shots": shots}}


def test_good_spot_passes():
    rows = m.check(_st([{"t": 1, "team": "A", "x_m": 96.0, "y_m": 32.0, "distance_m": 10.0}]))
    assert rows[0]["faults"] == []


def test_own_half_and_far_flagged():
    rows = m.check(_st([{"t": 1, "team": "B", "x_m": 96.0, "y_m": 32.0, "distance_m": 96.0}]))
    assert "in own half" in rows[0]["faults"] and "over 40 m" in rows[0]["faults"]


def test_wrong_distance_and_off_pitch_flagged():
    rows = m.check(_st([{"t": 1, "team": "A", "x_m": 110.0, "y_m": 32.0, "distance_m": 2.0}]))
    f = " ".join(rows[0]["faults"])
    assert "off the pitch" in f and "distance" in f


def test_no_spot_skipped():
    assert m.check(_st([{"t": 1, "team": "A", "x_m": None, "y_m": None}])) == []


def test_repo_vallentuna_spots_clean():
    p = os.path.join(REPO, "results/volume/runs/matches/p15u-vs-vallentuna-2026-10-03-6cce/stats.json")
    if not os.path.exists(p):
        return
    rows = m.check(json.load(open(p)))
    assert len(rows) >= 11 and not [r for r in rows if r["faults"]]
