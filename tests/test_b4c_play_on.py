"""B4c: the spare-ball rule (B4b) only while play is on. During play the picker leaves a ball lying at a recurring
off-pitch spot; in a stoppage (no ball moving on the pitch for a few seconds) a ball taken from that spot is followed."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import ball as BL

FPS, L, W = 10.0, 106.0, 64.0

def test_moving_ball_needs_smooth_motion_on_the_pitch():
    n = 40; Mb = np.full((n, 2), np.nan)
    Mb[0:10] = [[20 + 0.6 * k, 30] for k in range(10)]          # rolling ball, 6 m/s
    Mb[10:20] = [[50, 30] if k % 2 else [20, 30] for k in range(10)]   # jumping between guesses
    Mb[20:30] = [[-6 - 0.6 * k, 30] for k in range(10)]          # moving, but off the pitch
    Mb[30:40] = [[40, 30]] * 10                                  # still on the pitch
    mv = BL.moving_ball(Mb, FPS, L, W, disp_m=1.5)
    assert mv[3:10].all() and not mv[10:].any()
    inside = BL.moving_ball(np.array([[30 + 0.6 * k, 0.5] for k in range(10)]), FPS, L, W, margin=-2.0, disp_m=1.5)
    assert not inside.any()                                      # along the touchline, not 2 m inside

def test_play_on_mask_looks_back():
    mv = np.zeros(100, bool); mv[10] = True
    p = BL.play_on_mask(mv, FPS, 2.0)
    assert not p[:10].any() and p[10:31].all() and not p[31:].any()

def _scene():
    n = int(85 * FPS); H = [np.diag([10.0, 10.0, 1.0])] * n     # pixels = 10 x metres
    cands = {}; rng = np.random.default_rng(0)   # calibration jitter ~0.4 m like SFK-BP (a perfectly still guess
    for i in range(n):                            # with nobody near is the static-clutter rule's job, not this one)
        t = i / FPS; c = []; jx, jy = rng.normal(0, 4.0, 2)
        if t < 30 or t >= 52:                                    # match ball in play, back and forth at 6 m/s
            x = 10 + abs((0.6 * i) % 160 - 80); c.append((10 * x, 300.0, 0.5))
        if 3 <= t < 10 or 72 <= t < 79: c.append((-40.0 + jx, 340.0 + jy, 0.9))   # spare ball behind the goal during play (visits >= 30 s apart)
        if 36 <= t < 46: c.append((-40.0 + jx, 340.0 + jy, 0.9))           # stoppage: ball out of view, keeper goes to the spare ball
        if 46 <= t < 52: c.append((-40.0 + 8 * (t - 46), 340.0, 0.9))  # ... and carries it onto the pitch
        cands[i] = c
    return n, H, cands

def test_spare_dropped_in_play_followed_in_stoppage():
    n, H, cands = _scene()
    spot = lambda b, a, z: sum(1 for i, p in b.items() if a <= i / FPS < z and abs(p[0] + 40) < 15 and abs(p[1] - 340) < 15)
    always = BL.pick_v2(cands, H, L, W, fps=FPS, recur_r=2.0, log=lambda *a: None)
    gated = BL.pick_v2(cands, H, L, W, fps=FPS, recur_r=2.0, recur_play_s=4.0, recur_move_m=1.5, log=lambda *a: None)
    assert spot(always, 0, 46) == 0 and spot(gated, 3, 10) == 0 and spot(gated, 72, 79) == 0   # in play: left alone
    assert spot(gated, 36, 46) >= 0.9 * 10 * FPS                                              # stoppage: followed
    assert BL.pick_v2(cands, H, L, W, fps=FPS, log=lambda *a: None) != gated                   # and the rule did something
