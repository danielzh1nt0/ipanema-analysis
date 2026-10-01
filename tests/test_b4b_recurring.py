"""B4b: a ball lying still at the same off-pitch spot in two separate visits (spare ball behind the goal) is dropped;
a ball lying still off the pitch once (match ball before a restart) and still spots on the pitch are kept."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL

FPS, L, W = 10.0, 106.0, 64.0

def _rows(n, spots):
    """spots: list of (x_m, y_m, [(start_s, end_s), ...]) -> per frame rows (mx, my, conf, x, y, near, on)"""
    C = [[] for _ in range(n)]; rng = np.random.default_rng(0)
    for x, y, visits in spots:
        for a, b in visits:
            for i in range(int(a * FPS), int(b * FPS)):
                jx, jy = rng.normal(0, 0.4, 2)          # calibration jitter ~0.5 m, like SFK-BP
                C[i].append((x + jx, y + jy, 0.6, 0.0, 0.0, 1.0, False))
    return C

def test_spare_ball_recurs_restart_ball_does_not():
    n = int(120 * FPS)
    C = _rows(n, [(-4.0, 34.0, [(10, 16), (70, 76)]),      # spare ball: two visits 60 s apart, players near (near=1)
                  (10.5, -2.5, [(40, 47)]),                # match ball before a throw-in: one visit
                  (-4.0, 10.0, [(20, 21), (90, 91)])])     # passes through twice but never still for long
    s = BL.recurring_spots(C, FPS, L, W)
    assert len(s) == 1 and abs(s[0][0] + 4) < 1 and abs(s[0][1] - 34) < 1

def test_on_pitch_spots_never_count():
    n = int(120 * FPS)
    C = _rows(n, [(53.0, 32.0, [(10, 16), (70, 76)])])     # centre spot, kick-offs
    C = [[(r[0], r[1], r[2], r[3], r[4], r[5], True) for r in c] for c in C]
    assert BL.recurring_spots(C, FPS, L, W) == []

def test_picker_leaves_the_spare_ball():
    n = int(90 * FPS); Hm = np.diag([10.0, 10.0, 1.0]); H = [Hm] * n          # pixels = 10 x metres
    cands = {}
    for i in range(n):
        t = i / FPS; c = []
        if 5 <= t < 12 or 65 <= t < 72: c.append((-40.0, 340.0, 0.9))       # spare ball behind the goal, confident
        c.append((300.0 + 2 * i, 300.0, 0.5))                               # the match ball, moving, less confident
        cands[i] = c
    def on_spare(b): return sum(1 for i, p in b.items() if abs(p[0] + 40) < 5 and abs(p[1] - 340) < 5)
    old = BL.pick_v2(cands, H, L, W, fps=FPS, log=lambda *a: None)
    new = BL.pick_v2(cands, H, L, W, fps=FPS, recur_r=2.0, log=lambda *a: None)
    assert on_spare(old) > 0 and on_spare(new) == 0
    assert all(abs(new[i][0] - (300 + 2 * i)) < 1 for i in range(n) if i in new)
