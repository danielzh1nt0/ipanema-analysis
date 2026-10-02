import numpy as np
from ipanema import tracking as TR


def test_still_person_by_the_line_is_dropped_but_a_waiting_player_is_kept():
    fps = 30.0; per = {}
    for k in range(120):
        rows = [[1, "A", np.array([50.0, 0.8 + 0.01 * np.sin(k)]), np.array([0.0, 0.0]), (0, 0, 10, 20), False],     # coach, still, 0.8 m inside the line
                [2, "A", np.array([40.0 + 0.05 * k, 1.0]), np.array([0.0, 0.0]), (0, 0, 10, 20), False],              # player jogging along the touchline (moves 6 m)
                [3, "B", np.array([30.0, 30.0]), np.array([0.0, 0.0]), (0, 0, 10, 20), False]]                        # still, but mid-pitch
        per[k] = rows
    out, _ = TR.clean(per, 106.0, 64.0, fps, log=lambda *a: None)
    ids = {r[0] for k in out for r in out[k]}
    assert 1 not in ids and 2 in ids and 3 in ids
