"""N0-23c (23 Oct): the app's pass-direction figures (stats.json teams[].forward_pass_share_pct, progressive_passes and
per-player progressive_passes) come from analytics.passes: gain = receiver's spot at his first touch - passer's spot at
his last touch, towards the team's attacking goal; > 3 m forward, < -3 m backward, else sideways; progressive = gain
>= 10 m AND completed. Checked on Metrica pro data (tools/n0_23c_progressive.py). Also: no test may leave a fake module
in sys.modules (a leftover fake 'ultralytics' from two tests broke tests/test_tracking_batch.py whenever the real
ultralytics + torch were installed, i.e. on a GPU box, while the nightly runner without torch never saw it)."""
import os, sys, re, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import analytics as AN

FPS = 25


def _play(touches, n=None):
    """touches: [(player id, team, x, first frame, last frame[, y])]; every player stands still at (x, y = 34); the ball
    sits on whoever is in a touch window -> per, frames_ as analytics.passes reads them"""
    touches = [t if len(t) == 6 else (*t, 34.0) for t in touches]
    n = n or max(t[4] for t in touches) + 5
    per = [[[pid, tm, np.array([x, y]), None, None, False] for pid, tm, x, _, _, y in touches] for _ in range(n)]
    frames_ = []
    for k in range(n):
        c = next(((pid, tm) for pid, tm, x, a, b, y in touches if a <= k <= b), None)
        frames_.append({"frame": k, "carrier": c[0] if c else None, "team": c[1] if c else None})
    return per, frames_


def test_kind_and_progressive_attacking_left():
    # A attacks LEFT: 1 (x 60) -> 2 (x 48) gains 12 m = forward + progressive; 2 -> 3 (x 50) loses 2 m = sideways;
    # (3 stands 6 m wide: passes under 5 m are not passes); 3 -> 4 (x 62) loses 12 m = backward; 4 -> 9 of B (x 40): forward 22 m but lost -> not progressive
    t = [(1, "A", 60.0, 0, 29), (2, "A", 48.0, 40, 69), (3, "A", 50.0, 80, 109, 40.0), (4, "A", 62.0, 120, 149), (9, "B", 40.0, 160, 199)]
    per, fr = _play(t)
    ps, _ = AN.passes(per, fr, [], {}, {"A": False, "B": True}, FPS)
    got = [(p["from"], p["to"], p["kind"], p["progressive"], p["completed"], p["gain_m"]) for p in ps]
    assert got == [(1, 2, "forward", True, True, 12.0), (2, 3, "sideways", False, True, -2.0),
                   (3, 4, "backward", False, True, -12.0), (4, 9, "forward", False, False, 22.0)]


def test_progressive_threshold_is_10m():
    for x2, prog in ((50.0, True), (49.9, False)):                                 # A attacks right: 40 -> 50 = exactly 10 m counts, 9.9 m not
        per, fr = _play([(1, "A", 40.0, 0, 29), (2, "A", x2, 40, 69)])
        ps, _ = AN.passes(per, fr, [], {}, {"A": True, "B": False}, FPS)
        assert len(ps) == 1 and ps[0]["progressive"] is prog and ps[0]["kind"] == "forward"


def test_metrica_truth_direction_and_kind():
    import n0_23c_progressive as NQ
    P = [{"team": "A", "period": 1, "f0": 10, "f1": 20, "x0": 0.50, "y0": 0.5, "x1": 0.40, "y1": 0.5, "sub": ""},
         {"team": "B", "period": 1, "f0": 30, "f1": 40, "x0": 0.50, "y0": 0.5, "x1": 0.52, "y1": 0.5, "sub": ""},
         {"team": "A", "period": 2, "f0": 50, "f1": 60, "x0": 0.50, "y0": 0.5, "x1": 0.30, "y1": 0.5, "sub": ""}]
    T = NQ.truth_rows(P, 1, home_right=False)                                       # Home (A) attacks left in period 1
    assert [(r["team"], r["kind"], r["progressive"]) for r in T] == [("A", "forward", True), ("B", "sideways", False)]
    assert abs(T[0]["gain_m"] - 10.5) < 1e-9
    s = NQ.summary(T); assert s["A"] == {"n": 1, "forward_pct": 100.0, "progressive": 1}


def test_metrica_match_pairs_by_team_and_time():
    import n0_23c_progressive as NQ
    T = [{"team": "A", "f0": 125, "gain_m": 12.0, "kind": "forward", "progressive": True}]
    ours = [{"team": "B", "t": 1.0, "completed": True, "gain_m": 12.0, "kind": "forward", "progressive": True},
            {"team": "A", "t": 1.0, "completed": True, "gain_m": 9.0, "kind": "forward", "progressive": False}]
    m = NQ.match(T, ours, f0=100)
    assert m["n"] == 1 and m["kind_agree"] == 1 and m["prog_truth"] == 1 and m["prog_ours"] == 0 and m["gain_err_median_m"] == -3.0


def test_no_test_leaves_a_fake_module_behind():
    bad = []
    for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if re.search(r"sys\.modules\.setdefault\(|sys\.modules\[[^\]]+\]\s*=(?!=)", line) and "sitecustomize" not in line and "write_text" not in line:
                bad.append(f"{os.path.basename(f)}:{i}")
    assert not bad, f"use monkeypatch.setitem(sys.modules, ...) so the fake is removed after the test: {bad}"
