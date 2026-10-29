"""N0-29c (29 Oct): a player's top speed (stats.json players[].top_speed_ms) was the fastest single row speed, so feet
jitter and jumps to another person set it: 9.9 m/s on every Metrica player cut like our tracks (truth 7.2-8.1), 8+ m/s
on half the app clips' tracks. Now: the best 1-s average speed over time he was seen the whole time (one seen run, a raw
> 3 m jump in <= 3 frames splits it). IPANEMA_TOP_SPEED=old = old. High-intensity: a held-for-1-s rule was tried and
scored worse (15-35% of the truth), so it stays the old way (IPANEMA_HI=old default)."""
import os, sys, subprocess
import numpy as np, pytest

from ipanema import analytics as AN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25.0


def _track(segments, jitter=0.0, seed=0):
    """[(first frame, last frame, (x0, y0), (vx, vy) m/s)] -> ks, P"""
    ks, P = [], []
    for a, b, p0, v in segments:
        for k in range(a, b + 1):
            ks.append(k); P.append([p0[0] + v[0] * (k - a) / FPS, p0[1] + v[1] * (k - a) / FPS])
    P = np.array(P, float) + np.random.default_rng(seed).normal(0, jitter, (len(ks), 2))
    return np.array(ks), P


def _stats(ks, P):
    n = int(ks[-1]) + 1; per = {k: [] for k in range(n)}
    for k, p in zip(ks, P): per[int(k)].append([1, "A", np.asarray(p), np.zeros(2), None, False])
    tracks = {1: {int(k): np.asarray(p) for k, p in zip(ks, P)}}
    frames_ = [{"carrier": None, "team": None, "pressure_m": None, "near_opps": 0} for _ in range(n)]
    return AN.stats(per, frames_, [], [], tracks, np.full(n, 2), FPS, 105.0, 68.0, {"A": True, "B": False})["players"][0]


def test_steady_run_reads_its_speed():
    ks, P = _track([(0, 99, (10, 30), (6, 0))])
    assert AN.top_speed(ks, P, FPS) == pytest.approx(6.0, abs=0.1)


def test_jitter_on_a_still_player_is_not_a_sprint():
    ks, P = _track([(0, 499, (40, 30), (0, 0))], jitter=0.35)
    assert AN.top_speed(ks, P, FPS) < 2.5


def test_box_slide_on_a_still_player():
    ks = np.arange(250); P = np.tile([40.0, 30.0], (250, 1)); P[100:106, 0] += 2.5   # feet point slides 2.5 m for 0.24 s
    assert AN.top_speed(ks, P, FPS, old=True) > 3.0
    assert AN.top_speed(ks, P, FPS) < 1.6


def test_jump_to_another_person_is_not_a_speed():
    ks = np.arange(100); P = np.c_[10 + 2 * ks / FPS, np.full(100, 30.0)]; P[50:, 0] += 4   # 4 m jump between frames 49 and 50
    assert AN.top_speed(ks, P, FPS, old=True) > 6.5
    assert AN.top_speed(ks, P, FPS) == pytest.approx(2.0, abs=0.1)


def test_burst_shorter_than_a_second_does_not_count_alone():
    ks, P = _track([(0, 49, (10, 30), (3, 0)), (50, 59, (11.96 + 0.12, 30), (8, 0))])   # 0.4 s at 8 m/s, then the track ends
    assert AN.top_speed(ks, P, FPS) < 5.0


def test_gap_splits_the_window():
    ks, P = _track([(0, 14, (10, 30), (7, 0)), (40, 54, (20, 30), (7, 0))])            # two 0.6 s runs at 7 m/s, gap between
    assert AN.top_speed(ks, P, FPS) == 0.0                                            # never seen for a whole second


def test_stats_uses_new_and_old_switch(monkeypatch):
    ks = np.arange(100); P = np.c_[10 + 2 * ks / FPS, np.full(100, 30.0)]; P[50:, 0] += 4
    assert _stats(ks, P)["top_speed_ms"] == pytest.approx(2.0, abs=0.1)
    monkeypatch.setattr(AN, "TOP_SPEED", "old")
    assert _stats(ks, P)["top_speed_ms"] > 6.5


def test_high_intensity_stays_old_by_default():
    assert AN.HI_MODE == "old" or os.environ.get("IPANEMA_HI")
    ks, P = _track([(0, 99, (10, 30), (6.5, 0))])
    assert AN.high_intensity(ks, P, FPS, old=False) == pytest.approx(AN.high_intensity(ks, P, FPS, old=True), rel=0.1)


def test_lab_dry_run():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_29c_speed.py"), "--dry"], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "sus1" in r.stdout
