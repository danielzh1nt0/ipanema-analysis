"""N0-25c (25 Oct): the 'shape over time' timeline (stats.metrics.shape_timeline, read by the Phases ribbon / shape over
time chart) took block length / width from the players in the picture only (>= 3). On Metrica pro data seen through our
app clips' camera views each second read ~0.6-0.7x the truth (median error 10-12 m length, 12-14 m width). Now, with the
pipeline rows, length / width come from analytics.held_points (last seen spot kept 5 s, >= 5 points) - the same rule as
the Shape tab's team medians since N0-22e: ~1.0x, 3-4 m. Line height unchanged. IPANEMA_TIMELINE_HOLD=0 = old."""
import os, sys, json, subprocess
import numpy as np, pytest

from ipanema import analytics as AN, metrics as ME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 10.0
L, W = 105.0, 68.0
AR = {"A": True, "B": False}


def _row(tid, team, x, y, gk=False):
    return [tid, team, np.array([x, y], float), np.zeros(2), None, gk]


def _camera_follows(n=300):
    """for 2 s of every 3 the two deepest of team A's 7 outfield players are out of the picture"""
    per = {}
    for k in range(n):
        out = (k % 30) >= 10
        per[k] = [_row(0, "A", 2, 34, gk=True)] + [_row(i, "A", x, 30 + i) for i, x in enumerate((20, 30, 40, 50, 60, 70, 75), start=1) if not (out and i in (1, 2))]
    return per


def test_held_length_and_old_switch(monkeypatch):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", 5.0)
    per = _camera_follows(); sh = AN.shapes(per, L, FPS)
    old = ME.shape_timeline(sh, FPS, AR, L)                                    # no rows given: the old timeline
    new = ME.shape_timeline(sh, FPS, AR, L, per=per, W=W)
    off = ME.shape_timeline(sh, FPS, AR, L, per=per, W=W, hold=False)          # IPANEMA_TIMELINE_HOLD=0
    assert off == old
    assert len(new["A"]) == len(old["A"]) == 30
    lo = [e["length"] for e in old["A"]]; ln = [e["length"] for e in new["A"]]
    assert np.median(lo) == pytest.approx(35.0) and np.median(ln) == pytest.approx(55.0)   # whole team 20..75
    assert all(abs(a - b) < 1e-9 for a, b in zip([e["line_height"] for e in old["A"]], [e["line_height"] for e in new["A"]]))  # line unchanged
    assert all(e["n_held"] == 7 for e in new["A"][1:])
    assert all(e["length"] is None and e["n"] == 0 for e in new["B"])          # B never seen


def test_timeline_matches_the_card(monkeypatch):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", 5.0)
    per = _camera_follows(); n = len(per)
    tracks = {}
    for k in range(n):
        for r in per[k]: tracks.setdefault(r[0], {})[k] = r[2]
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    st = AN.stats(per, frames_, [], [], tracks, np.full(n, 2), FPS, L, W, AR)
    card = next(t for t in st["teams"] if t["team"] == "A")
    tl = ME.shape_timeline(AN.shapes(per, L, FPS), FPS, AR, L, per=per, W=W)
    assert np.median([e["length"] for e in tl["A"]]) == pytest.approx(card["block_length_median_m"])
    assert np.median([e["width"] for e in tl["A"]]) == pytest.approx(card["block_width_median_m"])


def test_too_few_points_gives_null_length_but_keeps_line(monkeypatch):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", 5.0)
    per = {k: [_row(1, "A", 30, 30), _row(2, "A", 60, 30), _row(3, "A", 50, 20)] for k in range(40)}   # 3 seen, nobody held
    sh = AN.shapes(per, L, FPS); tl = ME.shape_timeline(sh, FPS, AR, L, per=per, W=W)
    assert all(e["length"] is None and e["width"] is None for e in tl["A"])
    assert all(e["line_height"] == 30.0 for e in tl["A"])                     # line height from the seen players, as before
    old = ME.shape_timeline(sh, FPS, AR, L)
    assert all(e["length"] == 30.0 for e in old["A"])                          # old: a 'block' of 3 people


def test_shape_hold_off_keeps_old(monkeypatch):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", 0.0)                               # IPANEMA_SHAPE_HOLD_S=0: cards and timeline both old
    per = _camera_follows(); sh = AN.shapes(per, L, FPS)
    assert ME.shape_timeline(sh, FPS, AR, L, per=per, W=W) == ME.shape_timeline(sh, FPS, AR, L)


def test_compute_passes_the_rows(monkeypatch):
    monkeypatch.setattr(AN, "SHAPE_HOLD_S", 5.0)
    seen = {}
    orig = ME.shape_timeline
    def spy(*a, **k): seen.update(k); return orig(*a, **k)
    monkeypatch.setattr(ME, "shape_timeline", spy)
    per = _camera_follows(60); sh = AN.shapes(per, L, FPS); n = len(per)
    state = np.full(n, 2); ballm = {}; bspeed = np.zeros(n)
    try:
        ME.compute(state, ballm, bspeed, FPS, L, W, AR, [], [], [], [], sh, per=per)
    except Exception:
        pass                                                                    # other metrics may want more inputs; only the call matters
    assert seen.get("per") is per and seen.get("W") == W


def test_env_switch():
    code = "import os; os.environ['IPANEMA_TIMELINE_HOLD']='0'; from ipanema import metrics as ME; print(ME.TIMELINE_HOLD)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_metrica_tool_dry_run():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_25c_timeline.py"), "--dry"], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert set(d) >= {"old", "new", "card_median", "truth_median"}
    assert d["new"]["length"]["covered"] > 0.5
