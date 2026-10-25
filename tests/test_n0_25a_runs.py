"""N0-25a: the app's runs (metrics.runs) read the sprint speed from smoothed positions, not raw frame steps."""
import os, sys, subprocess, numpy as np
from ipanema import metrics as M

FPS = 25
AR = {"A": True, "B": False}


def _per(xs, ys=None, tid=7, team="A", start=0, drop=()):
    ys = [30.0] * len(xs) if ys is None else ys
    return {start + i: ([] if i in drop else [(tid, team, np.array([x, y]))]) for i, (x, y) in enumerate(zip(xs, ys))}


def _sprint_track(rng=None, jit=0.0, v=7.0, still=2.0, run=3.0):
    """stand still, sprint at v m/s for `run` s, stand still"""
    n0, n1 = int(still * FPS), int(run * FPS); x = [20.0] * n0 + [20 + v * (i + 1) / FPS for i in range(n1)]
    x += [x[-1]] * n0; x = np.array(x)
    y = np.full(len(x), 30.0)
    if rng is not None: x = x + rng.normal(0, jit, len(x)); y = y + rng.normal(0, jit, len(x))
    return x, y


def test_sprint_with_our_jitter_is_found_once_old_rule_misses_it():
    rng = np.random.default_rng(0); x, y = _sprint_track(rng, jit=0.3)
    per = _per(x, y); st = np.zeros(len(x), int)
    new = M.runs(per, FPS, st, AR, smooth=True)
    assert len(new) == 1 and 6.0 <= new[0]["speed_ms"] <= 8.0 and 2.0 <= new[0]["t"] <= 2.6 and new[0]["with_ball"] is True
    assert 14 <= new[0]["distance_m"] <= 23
    assert len(M.runs(per, FPS, st, AR, smooth=False)) != 1          # raw frame speed with 0.3 m jitter: none or broken pieces


def test_jogging_with_jitter_is_not_a_sprint():
    rng = np.random.default_rng(1); x, y = _sprint_track(rng, jit=0.3, v=3.5)
    assert M.runs(_per(x, y), FPS, np.zeros(len(x), int), AR, smooth=True) == []


def test_a_jump_to_another_person_breaks_the_run():
    # standing at x=20, then the track jumps to another standing player 6 m away (an id swap): smoothed over 0.8 s the
    # jump reads as 7.5 m/s for ~0.8 s, i.e. a 'sprint' (seen on the SFK-BP clip, results/qa/n0-25a)
    x = np.array([20.0] * 50 + [26.0] * 50)
    per = _per(x)
    assert M.runs(per, FPS, np.zeros(len(x), int), AR, smooth=True) == []
    assert M.runs(per, FPS, np.zeros(len(x), int), AR, smooth=True, jump_m=0) != []   # what the jump rule removes


def test_short_dip_is_merged_and_a_track_gap_splits_the_smoothing():
    x, y = _sprint_track(np.random.default_rng(2), jit=0.1, v=7.0, run=4.0)
    # a 0.2 s slow-down in the middle of the sprint (speed 4 m/s) -> still one run
    x2 = x.copy(); k = 50 + 50; x2[k:] -= np.minimum(np.arange(len(x2) - k), 5) * 0.6 / FPS * 5 / 5
    out = M.runs(_per(x2, y), FPS, np.zeros(len(x), int), AR, smooth=True, merge_s=0.5)
    assert len(out) == 1
    # a player standing still on both sides of a 2 s gap at different spots never 'sprints' across it
    xs = [20.0] * 50 + [35.0] * 50; per = _per(xs, drop=())
    per = {k: v for k, v in per.items()}; per = {k + (50 if k >= 50 else 0): v for k, v in per.items()}
    assert M.runs(per, FPS, np.zeros(200, int), AR, smooth=True) == []


def test_env_switch_restores_old_rule():
    code = "from ipanema import metrics as M; print(M.RUNS_SMOOTH)"
    env = dict(os.environ, IPANEMA_RUNS_SMOOTH="0", PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.strip() == "False"


def test_lab_dry_run():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "tools", "n0_25a_runs.py"), "--dry"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=root), timeout=300)
    assert r.returncode == 0 and '"old"' in r.stdout and '"new"' in r.stdout


def test_straight_line_gap_fill_is_not_a_sprint():
    # standing at x=20, a 1 s gap filled by fill_gaps as a straight line to x=27 (7 m/s), then standing: no run
    x = [20.0] * 50 + [20.0 + 7.0 * (i + 1) / FPS for i in range(25)] + [27.0 + 7.0 / FPS] * 50
    per = {k: [(7, "A", np.array([v, 30.0]), None, None, False) + (("filled",) if 50 <= k < 75 else ())] for k, v in enumerate(x)}
    assert M.runs(per, FPS, np.zeros(len(x), int), AR, smooth=True) == []
    per_unmarked = {k: [(7, "A", np.array([v, 30.0]))] for k, v in enumerate(x)}          # older inputs without the mark
    assert M.runs(per_unmarked, FPS, np.zeros(len(x), int), AR, smooth=True) == []
    assert len(M.runs(per_unmarked, FPS, np.zeros(len(x), int), AR, smooth=False)) == 1     # the old rule called it a sprint
