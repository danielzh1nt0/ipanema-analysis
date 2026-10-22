"""N0-22f (22 Oct): 'regained within 5 s' (counter-press success) / 'lost back in 5 s' in stats.json teams[]. Metrica check
(tools/n0_22f_regain.py): truth 17-28% of losses per half, the pipeline 2-19% with a perfect ball, 0-11% with our noise.
IPANEMA_REGAIN=withhold writes None for both (the app shows a dash); default 'on' = unchanged."""
import os, sys
import numpy as np
from ipanema import analytics as AN
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

FPS = 10


def _tv(k, lost_by, regained):
    won = "B" if lost_by == "A" else "A"
    return {"frame": k, "t": k / FPS, "frame_lost": k, "lost_by": lost_by, "won_by": won, "time_to_press": 1.0, "near_at_2s": 2,
            "regained_within_5s": regained, "lost_back_5s": regained, "time_to_forward_pass": None}


def _stats(monkeypatch, mode):
    monkeypatch.setattr(AN, "REGAIN", mode)
    n = 50; per = {k: [[1, "A", np.array([30.0, 30.0]), None, None, False], [2, "B", np.array([60.0, 30.0]), None, None, False]] for k in range(n)}
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    tracks = {1: {k: per[k][0][2] for k in range(n)}, 2: {k: per[k][1][2] for k in range(n)}}
    tvs = [_tv(5, "A", True), _tv(15, "A", False), _tv(25, "A", False), _tv(35, "B", False)]
    st = AN.stats(per, frames_, tvs, [], tracks, np.full(n, 2), FPS, 105.0, 68.0, {"A": True, "B": False})
    return {t["team"]: t for t in st["teams"]}


def test_default_unchanged(monkeypatch):
    t = _stats(monkeypatch, "on")
    assert t["A"]["regained_within_5s_pct"] == 33 and t["B"]["lost_back_5s_pct"] == 33   # 1 of A's 3 losses
    assert t["B"]["regained_within_5s_pct"] == 0 and "regain_withheld" not in t["A"]


def test_withhold(monkeypatch):
    t = _stats(monkeypatch, "withhold")
    for tm in ("A", "B"):
        assert t[tm]["regained_within_5s_pct"] is None and t[tm]["lost_back_5s_pct"] is None and t[tm]["regain_withheld"]
        assert t[tm]["losses"] in (1, 3) and t[tm]["pressed_within_2s_pct"] is not None   # the rest of the row stays


def test_truth_regains_from_events():
    import n0_22f_regain as R
    H, A = "Home", "Away"
    def e(team, typ, f0, f1=None): return {"team": team, "type": typ, "sub": "", "period": 1, "f0": f0, "f1": f1 or f0}
    ev = [e(H, "PASS", 0, 20), e(H, "BALL LOST", 30), e(A, "RECOVERY", 31), e(A, "PASS", 40, 50),   # Home loses ...
          e(H, "RECOVERY", 100), e(H, "PASS", 110, 300),                                            # ... back after 70 frames = 2.8 s
          e(H, "BALL LOST", 400), e(A, "RECOVERY", 401), e(A, "PASS", 410, 600), e(H, "RECOVERY", 700)]   # back after 12 s
    T = R.truth_regains(ev, 1, 25.0)
    assert [(t["lost_by"], t["regained"], t["back_s"]) for t in T] == [("A", True, 2.8), ("A", False, 12.0)]
    assert T[1]["winner_chain_s"] == (600 - 401) / 25.0 and T[1]["winner_pass"]
    assert R.rate([True, False, False, False]) == 25.0 and R.rate([]) is None
