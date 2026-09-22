"""Automatic periods: warm-up is not match time, kick-offs are found by formation, half-time by the empty pitch."""
import numpy as np
from ipanema.periods import per_second, detect, veo_second_half_kickoff

def _match(fps=2.0, L=120.0, W=70.0):
    rng = np.random.RandomState(0); per = {}
    K1, HT0, HT1, END, TOTAL = 150, 2900, 3400, 5800, 6200
    for s in range(TOTAL):
        rows = []
        if s < K1:                                                  # warm-up: both teams all over the pitch
            for j in range(20): rows.append([j, "A" if j < 10 else "B", np.array([rng.uniform(5, 115), rng.uniform(5, 65)])])
        elif HT0 <= s < HT1 or s >= END:                            # half-time / after full time: a few people on the pitch
            for j in range(3): rows.append([j, "A", np.array([rng.uniform(5, 115), rng.uniform(5, 65)])])
        else:
            formation = (s - K1 < 5) or (0 <= s - HT1 < 5)              # the first seconds of each half: teams in their own halves
            for j in range(20):
                team = "A" if j < 10 else "B"
                if formation: x = rng.uniform(10, 58) if team == "A" else rng.uniform(62, 110)
                else: x = rng.uniform(5, 115)
                rows.append([j, team, np.array([x, rng.uniform(5, 65)])])
        for f in range(int(fps)): per[int(s * fps) + f] = rows
    return per, fps, (K1, HT0, HT1, END)

def test_detect_periods():
    per, fps, (K1, HT0, HT1, END) = _match()
    r = detect(per_second(per, fps, 120.0, 70.0))
    (a1, b1), (a2, b2) = r["periods"]
    assert abs(a1 - K1) <= 2 and abs(a2 - HT1) <= 2, r
    assert abs(b1 - HT0) <= 20 and abs(b2 - END) <= 20, r
    assert r["evidence"]["kickoff_1_by_formation"] and r["evidence"]["kickoff_2_by_formation"]

def test_veo_kickoff():
    lo, hi = veo_second_half_kickoff([2619, 4422, 5624], [57, 77], 3060)       # SFK-BP: goals 2-0 (57') and 2-1 (77')
    assert 3700 <= lo <= hi <= 3770, (lo, hi)
