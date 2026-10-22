"""N0-22c (22 Oct): per-player distance / high-intensity / top speed in analytics.stats smoothed a track's positions with
ONE rolling mean over all its rows, so positions from after a gap (player hidden, track stitched across up to 2.5 s)
leaked into the frames before it: a player standing still on both sides of a 2 s gap 'ran' 4.8 m, all at sprint speed
(top 6.25 m/s), and the real movement during a gap was never counted. Now: smoothing per seen run, and a gap of <= 2.5 s
counts as a straight line when that is 0.5-7 m/s. IPANEMA_RUN_SMOOTH=0 IPANEMA_GAP_LINE_S=0 = old numbers."""
import os, sys, subprocess
import numpy as np, pandas as pd, pytest

from ipanema import analytics as AN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30.0


def _track(segments):
    """[(first frame, last frame, (x0, y0), (vx, vy) m/s)] -> ks, P"""
    ks, P = [], []
    for a, b, p0, v in segments:
        for k in range(a, b + 1):
            ks.append(k); P.append([p0[0] + v[0] * (k - a) / FPS, p0[1] + v[1] * (k - a) / FPS])
    return np.array(ks), np.array(P, float)


def _stats(ks, P, team="A"):
    n = int(ks[-1]) + 1; per = {k: [] for k in range(n)}
    for k, p in zip(ks, P): per[int(k)].append([1, team, np.asarray(p), np.zeros(2), None, False])
    tracks = {1: {int(k): np.asarray(p) for k, p in zip(ks, P)}}
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    return AN.stats(per, frames_, [], [], tracks, np.full(n, 2), FPS, 105.0, 68.0, {"A": True, "B": False})["players"][0]


def test_still_both_sides_of_a_gap_is_no_run():
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (150, 239, (10, 30), (0, 0))])        # still, 2 s hidden, still at the same spot
    sp, step, hi = AN.track_motion(ks, P, FPS)
    assert step.sum() == 0 and hi.sum() == 0 and np.nanmax(sp) == 0
    sp, step, hi = AN.track_motion(ks, P, FPS, runs=False, gap_line_s=0)                 # old: positions leaked across the gap
    assert step.sum() == 0   # same spot: nothing to leak


def test_moved_during_gap_counts_straight_line_not_as_sprint():
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (150, 239, (15, 30), (0, 0))])        # 5 m in a 2 s gap = 2.5 m/s
    sp, step, hi = AN.track_motion(ks, P, FPS)
    assert step.sum() == pytest.approx(5.0, abs=0.01) and hi.sum() == 0 and np.nanmax(sp) == 0   # gap speed never top speed
    sp_o, step_o, hi_o = AN.track_motion(ks, P, FPS, runs=False, gap_line_s=0)
    assert step_o.sum() == pytest.approx(4.8, abs=0.2) and hi_o.sum() == pytest.approx(4.8, abs=0.2) and np.nanmax(sp_o) > 6   # the old fault
    p = _stats(ks, P)
    assert p["distance_m"] == pytest.approx(5.0, abs=0.1) and p["high_intensity_m"] == 0 and p["top_speed_ms"] == 0


def test_gap_line_limits():
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (120, 209, (40, 30), (0, 0))])        # 30 m in 1 s: another person
    assert AN.track_motion(ks, P, FPS)[1].sum() == 0
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (120, 209, (16.5, 30), (0, 0))])      # 6.5 m in 1 s: a sprint, counted as one
    _, step, hi = AN.track_motion(ks, P, FPS)
    assert step.sum() == pytest.approx(6.5, abs=0.01) and hi.sum() == pytest.approx(6.5, abs=0.01)
    assert AN.track_motion(ks, P, FPS, gap_max_ms=6.0)[1].sum() == 0
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (180, 269, (15, 30), (0, 0))])        # 3 s gap: too long to guess
    assert AN.track_motion(ks, P, FPS)[1].sum() == 0
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (150, 239, (10.5, 30), (0, 0))])      # 0.5 m in 2 s: standing (< 0.5 m/s)
    assert AN.track_motion(ks, P, FPS)[1].sum() == 0


def test_no_gap_unchanged():
    ks, P = _track([(0, 179, (20, 40), (4.0, 0))])                                     # 6 s at 4 m/s, never hidden
    new = AN.track_motion(ks, P, FPS); old = AN.track_motion(ks, P, FPS, runs=False, gap_line_s=0)
    for a, b in zip(new, old): np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))
    assert new[1].sum() == pytest.approx(4.0 * 179 / FPS, rel=0.08)                       # the centred mean shortens a run's ends a bit (old too)
    np.testing.assert_allclose(AN.smooth_runs(ks, P, 24), pd.DataFrame(P).rolling(24, center=True, min_periods=1).mean().values)


def test_smooth_runs_never_mixes_runs():
    ks, P = _track([(0, 29, (0, 0), (0, 0)), (40, 69, (50, 0), (0, 0))])
    S = AN.smooth_runs(ks, P, 24)
    assert (S[:30, 0] == 0).all() and (S[30:, 0] == 50).all()
    S = AN.smooth_runs(ks, P, 24, runs=False)
    assert S[29, 0] > 10                                                                 # old: the run before the gap is pulled 50 m away


def test_old_switch_through_stats(monkeypatch):
    ks, P = _track([(0, 89, (10, 30), (0, 0)), (150, 239, (15, 30), (0, 0))])
    monkeypatch.setattr(AN, "RUN_SMOOTH", False); monkeypatch.setattr(AN, "GAP_LINE_S", 0.0)
    p = _stats(ks, P)
    assert p["distance_m"] == pytest.approx(4.8, abs=0.2) and p["high_intensity_m"] > 4


@pytest.mark.parametrize("tool", ["n0_22c_run_smooth.py", "n0_22c_metrica.py"])
def test_tools_dry_run(tool):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), "--dry"], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    assert ("distance_m_A_new" in r.stdout) if "run_smooth" in tool else ("line_cap7" in r.stdout)
