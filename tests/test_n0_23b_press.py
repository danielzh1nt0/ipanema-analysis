"""N0-23b (23 Oct): the Metrica answer key for the counter-press figures (stats.json teams[].time_to_press_median_s,
pressed_within_2s_pct, near_at_2s_median) - tools/n0_23b_press.py. Truth: from the winner's RECOVERY, the first frame a
loser is within 2 m of the winner's carrier (the winner's player nearest the true ball, <= 2.5 m), until the winner's
chain ends; near_at_2s = losers within 5 m of the carrier 2 s after the win."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_23b_press as NP

FPS = 25


def _frames(closer_at_s=1.0, n_s=6):
    """Home #101 lost it; Away #201 has the ball at (50, 34) from frame 100; Home #102 walks in from 10 m and is within
    2 m from closer_at_s; Home #103 stands 4 m away (within 5 m), Home #104 far away."""
    fr = {}
    for k in range(0, int(n_s * FPS) + 100):
        t = max(0.0, (k - 100) / FPS)
        x102 = 50 + max(1.5, 10 - (10 - 1.5) * min(1.0, t / closer_at_s))
        ps = [(101, 30.0, 34.0), (102, x102, 34.0), (103, 50.0, 38.0), (104, 80.0, 10.0), (201, 50.0, 34.0)]
        fr[k] = (1, ps, (50.3, 34.0))
    return fr


def _events(end_at=None):
    ev = [{"team": "Home", "type": "BALL LOST", "sub": "", "period": 1, "f0": 98, "f1": 98, "x0": None, "x1": None},
          {"team": "Away", "type": "RECOVERY", "sub": "", "period": 1, "f0": 100, "f1": 100, "x0": None, "x1": None}]
    if end_at is not None:
        ev.append({"team": "Home", "type": "RECOVERY", "sub": "", "period": 1, "f0": end_at, "f1": end_at, "x0": None, "x1": None})
    return ev


def test_truth_time_to_press_and_near():
    T = NP.truth_press(_events(), _frames(closer_at_s=1.0), 1)
    assert len(T) == 1 and T[0]["won_by"] == "B"
    assert abs(T[0]["time_to_press"] - 1.0) <= 0.08           # #102 within 2 m after ~1 s
    assert T[0]["near_at_2s"] == 2                            # #102 (1.5 m) + #103 (4 m); #104 far


def test_truth_never_pressed_when_chain_ends_first():
    T = NP.truth_press(_events(end_at=100 + int(1.5 * FPS)), _frames(closer_at_s=3.0), 1)
    assert T[0]["time_to_press"] is None                      # Home won it back before #102 arrived


def test_summary_and_matching():
    T = [{"f_lost": 98, "f_won": 100, "won_by": "B", "time_to_press": 0.4, "near_at_2s": 1},
         {"f_lost": 500, "f_won": 510, "won_by": "A", "time_to_press": 3.0, "near_at_2s": 0}]
    ours = [{"frame": 105, "frame_lost": 99, "won_by": "B", "time_to_press": 0.6, "near_at_2s": 1},
            {"frame": 520, "frame_lost": 500, "won_by": "B", "time_to_press": 0.0, "near_at_2s": 2}]   # wrong team: no match
    s = NP.summ(T); assert s["within2_pct"] == 50.0 and s["median_s"] == 1.7
    m = NP.matched(T, ours, 0)
    assert m["n"] == 1 and m["agree_w2"] == 1 and abs(m["median_err_s"] - 0.2) < 1e-9


def test_view_frames_drops_players_outside_the_view():
    fr = _frames(); mask = np.zeros((106, 35), bool); mask[40:61, :] = True       # view: x 40-60 m of a 105 x 68 clip pitch
    out, share = NP.view_frames(fr, 1, [(mask, 50 / 105)], 105.0, 68.0, None, None, np.zeros((1, 2)), seed=0)
    ids = {i for i, _, _ in out[150][1]}
    assert 201 in ids and 103 in ids and 104 not in ids and 101 not in ids
    assert 0 < share < 1
