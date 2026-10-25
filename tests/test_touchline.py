"""TOUCHLINE (25 Oct): tracking.touchline_rows / clean() step 1c drop stretches of people standing by a side touchline
(spectators, staff behind the far line), keep wingers, throw-ins and anyone deeper in."""
import os, sys, subprocess, numpy as np
from ipanema import tracking as TR

FPS, L, W = 30.0, 106.0, 64.0
RNG = np.random.default_rng(0)


def row(t, x, y, team="A"):
    return [t, team, np.array([x, y]) + RNG.normal(0, 0.15, 2), np.array([0.0, 0.0]), (0, 0, 10, 20), False]


def scene(n=int(12 * FPS)):
    per = {}
    for k in range(n):
        s = k / FPS; rows = [
            row(1, 70.0 + 0.33 * s, 62.6),                             # spectator behind the far line, lands 1.4 m inside, drifts with the pan
            row(2, 20 + 4.0 * s, 62.0),                                 # winger running along the line at 4 m/s
            row(3, 50.0, 30.0, "B")]                                   # player standing mid-pitch
        # throw-in: stands on the line 4 s, then runs in
        rows.append(row(4, 40.0, 63.5) if s < 4 else row(4, 40.0 + 0.0 * s, 63.5 - 3.0 * (s - 4)))
        # id slide: spectator (8 s by the near line) and then the same id jumps to a player mid-pitch
        rows.append(row(5, 30.0, 1.2, "B") if s < 8 else row(5, 60.0 + 2 * (s - 8), 25.0, "B"))
        per[k] = rows
    return per


def test_rows_dropped_only_for_standing_touchline_stretches():
    per = scene(); dr = TR.touchline_rows(per, W, FPS)
    who = {t for _, t in dr}
    assert 1 in who and 2 not in who and 3 not in who and 4 not in who
    five = sorted(k for k, t in dr if t == 5)
    assert five and max(five) < 8 * FPS                      # only the spectator part of the slid id
    assert len(five) >= 7 * FPS


def test_slow_walker_along_the_line_kept_and_long_wait_dropped():
    per = {k: [row(1, 50 + 1.5 * k / FPS, 63.0), row(2, 10.0, 0.5)] for k in range(int(10 * FPS))}
    who = {t for _, t in TR.touchline_rows(per, W, FPS)}
    assert who == {2}


def test_gap_and_short_stretch_not_dropped():
    # standing by the line 5 s, gone 2 s, back 5 s: two stretches of 5 s, each under 6 s
    per = {k: ([row(1, 50.0, 63.0)] if not (5 * FPS <= k < 7 * FPS) else []) for k in range(int(12 * FPS))}
    assert not TR.touchline_rows(per, W, FPS)
    assert TR.touchline_rows(per, W, FPS, min_s=4.0)


def test_stretches_split_on_id_jump_and_fps_aware():
    ks = np.arange(20); P = np.c_[np.r_[np.zeros(10), np.full(10, 30.0)], np.zeros(20)]
    assert TR.track_stretches(ks, P, 30.0) == [(0, 10), (10, 20)]
    ks10 = np.r_[0:5, 9:14]; P10 = np.zeros((10, 2))          # 10 rows/s: 0.4 s gap is no break, 30 rows/s: 4 frames = 0.13 s
    assert TR.track_stretches(ks10, P10, 10.0) == [(0, 10)]
    assert TR.track_stretches(np.r_[0:5, 30:35], np.zeros((10, 2)), 30.0) == [(0, 5), (5, 10)]


def test_clean_switch(monkeypatch):
    monkeypatch.setenv("IPANEMA_TOUCHLINE_S", "0")
    out, _ = TR.clean(scene(), L, W, FPS, log=lambda *a: None)
    assert 1 in {r[0] for k in out for r in out[k]}
    monkeypatch.setenv("IPANEMA_TOUCHLINE_S", "6")
    out, _ = TR.clean(scene(), L, W, FPS, log=lambda *a: None)
    ids = {r[0] for k in out for r in out[k]}
    assert 1 not in ids and {2, 3, 4} <= ids


def test_lab_dry_run():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "tools/touchline_lab.py", "--dry"], cwd=root, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0 and "spectator rows 1500" in r.stdout
