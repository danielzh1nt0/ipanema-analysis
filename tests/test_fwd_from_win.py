"""N0-23a (23 Oct): 'forward pass within 3 s of winning it' (stats.json teams[].forward_within_3s_pct,
time_to_forward_pass_median_s) is timed from the winner's first touch after the loser's last control, not from the frame
the counted state turns to the winner (that frame often comes AFTER the first pass: the receiver is the first player the
state trusts). IPANEMA_FWD_FROM_WIN=0 = old. Also: tools/metricalab.ours takes the attack direction like run.py (it was a
fixed A ->, backwards in 2 of Metrica's 4 halves)."""
import os, sys
import numpy as np
from ipanema import possession as P
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

FPS = 10


def _scene():
    """B (id 2) holds the ball 0-4.9 s, loses it; A's #1 touches it at 5.0 s (state still loose), passes 10 m forward to A's
    #3, who has it from 6.0 s; the counted state turns to A only at 6.0 s. A attacks right."""
    n = 150; per = {}; frames_ = []; state = []; ballm = {}
    for k in range(n):
        t = k / FPS
        rows = [[1, "A", np.array([40.0, 30.0]), None, None, False], [3, "A", np.array([50.0, 30.0]), None, None, False],
                [2, "B", np.array([38.0, 32.0]), None, None, False]]
        per[k] = rows
        if t < 5.0: c, tm, pos, st, b = 2, "B", [38.0, 32.0], 1, np.array([38.0, 32.0])
        elif t < 5.5: c, tm, pos, st, b = 1, "A", [40.0, 30.0], 2, np.array([40.0, 30.0])
        elif t < 6.0: c, tm, pos, st, b = None, None, None, 2, np.array([40.0 + (t - 5.5) * 20, 30.0])
        else: c, tm, pos, st, b = 3, "A", [50.0, 30.0], 0, np.array([50.0, 30.0])
        frames_.append({"frame": k, "carrier": c, "team": tm, "pos": pos, "pressure_m": 5.0, "near_opps": 0})
        state.append(st); ballm[k] = b
    return per, frames_, state, ballm


def _tv(monkeypatch, on):
    monkeypatch.setattr(P, "FWD_FROM_WIN", on)
    per, frames_, state, ballm = _scene()
    tvs = P.turnovers(per, frames_, state, ballm, FPS, {"A": True, "B": False})
    assert len(tvs) == 1 and tvs[0]["won_by"] == "A"
    return tvs[0]


def test_first_pass_counted_from_first_touch(monkeypatch):
    tv = _tv(monkeypatch, True)
    assert tv["time_to_forward_pass"] == 1.0                     # #1's touch at 5.0 s -> #3 has it at 6.0 s
    assert tv["t_won"] == 6.1                                    # the win frame for the other figures is unchanged (first slow-ball frame)


def test_old_misses_it(monkeypatch):
    tv = _tv(monkeypatch, False)
    assert tv["time_to_forward_pass"] is None                     # old: search starts at #3, the pass is before it


def test_backward_pass_not_counted(monkeypatch):
    monkeypatch.setattr(P, "FWD_FROM_WIN", True)
    per, frames_, state, ballm = _scene()
    tvs = P.turnovers(per, frames_, state, ballm, FPS, {"A": False, "B": True})   # A attacks left: the same pass is backward
    assert tvs[0]["time_to_forward_pass"] is None


def test_metricalab_direction_from_players():
    import metricalab as ML
    # two short halves, A deep on the right (defends right -> attacks left); no keepers flagged in Metrica rows
    fr = {}
    for k in range(1, 400):
        ps = [(100 + i, 70.0 + i, 10.0 + 5 * i) for i in range(6)] + [(200 + i, 35.0 - i, 10.0 + 5 * i) for i in range(6)]
        fr[k] = (1, ps, (52.0, 34.0))
    o = ML.ours(fr, 1, False)
    assert o["attack_right"] == {"A": False, "B": True}
