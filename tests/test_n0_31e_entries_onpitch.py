"""N0-31e (31 Oct): final-third entries and field tilt skip balls off the pitch (metrics.field_tilt_and_entries with W).
On the app copies 34 of 687 entries had the ball beyond 2 m of a touchline (a ball in the air projects beyond the far
line, a pick on the fence); the app's 'Where did they get in?' drew them off the pitch."""
import numpy as np
from ipanema import metrics as M

L, W, FPS = 106.0, 64.0, 10.0


def _field(pts, team=0, right=True, **kw):
    state = np.full(len(pts), team); ballm = {k: p for k, p in enumerate(pts) if p is not None}
    tm = "A" if team == 0 else "B"; other = "B" if tm == "A" else "A"
    return M.field_tilt_and_entries(state, ballm, {}, FPS, L, {tm: right, other: not right}, *M.ENTRY_REARM, hold_s=3.0, **kw)[tm]


def test_entry_off_pitch_is_dropped():
    pts = [(60, 30), (66, 30), (69, -20), (72, -25), (74, -22)]      # crosses the line while projected 20 m beyond the far line
    assert _field(pts, onpitch=False)["entries_count"] == 1                            # old rule: counted, y_m -25
    assert _field(pts, onpitch=False)["final_third_entries"][0]["y_m"] == -25.0
    assert _field(pts, W=W, onpitch=True)["entries_count"] == 0


def test_entry_on_pitch_kept_and_skip_bridges_one_bad_frame():
    pts = [(60, 30), (66, 30), (69, 30), (71, 488), (72, 31), (74, 31)]   # one wild frame on the line
    r = _field(pts, W=W, onpitch=True)
    assert r["entries_count"] == 1 and r["final_third_entries"][0]["y_m"] == 31.0
    assert _field(pts, onpitch=False)["final_third_entries"][0]["y_m"] == 488.0


def test_margin_keeps_ball_just_out():
    pts = [(60, 63), (66, 64.5), (69, 65.5), (72, 65.9)]             # on the near line, within 2 m
    assert _field(pts, W=W, onpitch=True)["entries_count"] == 1


def test_tilt_ignores_off_pitch_frames():
    pts = [(30, 30)] * 6 + [(80, -30)] * 4                            # 4 frames 'in the final third' but 30 m off the pitch
    assert _field(pts, onpitch=False)["field_tilt_pct"] == 40
    assert _field(pts, W=W, onpitch=True)["field_tilt_pct"] == 0


def test_no_width_is_old_rule():
    pts = [(60, 30), (66, 30), (69, -20), (72, -25)]
    assert _field(pts) == _field(pts, onpitch=False)


def test_compute_passes_width(monkeypatch):
    seen = {}
    def fake(*a, **k): seen.update(k); return {}
    monkeypatch.setattr(M, "field_tilt_and_entries", fake)
    try:
        M.compute(np.zeros(3, int), {}, {}, FPS, L, W, {"A": True, "B": False}, [], [], [], [], {})
    except Exception:
        pass
    assert seen.get("W") == W


def test_frames_tool_plan_picks_nearest_ball():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import n0_31e_frames as F
    fr = [{"t": 0.1 * i, "ball": {"px": [i, 2 * i]}} for i in range(40)]
    P = F.plan([{"id": "e00", "match": "m", "team": "A", "attacks": "right", "t": 3.0}], lambda mid: ({}, fr))
    (l1, t1, p1), (l2, t2, p2), (l3, t3, p3) = P[0]["tiles"]
    assert (l1, l3) == ("t-1.0", "t") and abs(t1 - 2.0) < 1e-6 and p1 == [20, 40] and p3 == [30, 60]
    assert F.ball_at(fr, [f["t"] for f in fr], 9.0)[1] is None                          # no frame within 0.2 s
