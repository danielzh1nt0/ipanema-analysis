"""4 Oct: the goalmouth keeper rule must not repaint outfield players or flag several keepers per end."""
import numpy as np
from ipanema import tracking as TR

def test_attacker_in_goalmouth_keeps_team():
    L, W, fps, n = 106.0, 64.0, 10.0, 200
    per = {}
    for k in range(n):
        rows = [[1, "A", np.array([3.0, 32.0]), np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), False],     # A keeper, left goal
                [2, "B", np.array([4.0, 30.0]), np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), False],     # B attacker in the same goalmouth
                [3, "B", np.array([5.0, 34.0]), np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), False]]     # another B attacker
        rows += [[10 + i, "A", np.array([20.0 + i, 20.0 + i]), np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), False] for i in range(5)]
        rows += [[20 + i, "B", np.array([70.0 + i, 20.0 + i]), np.array([0.0, 0.0]), np.array([0, 0, 1, 1]), False] for i in range(5)]
        per[k] = rows
    try: out, info = TR.clean(per, L, W, fps, log=lambda *a: None)
    except TypeError: out, info = TR.clean(per, L, W, fps)
    rows = out[100]; team = {r[0]: r[1] for r in rows}; gk = {r[0]: r[5] for r in rows}
    assert team.get(2) == "B" and team.get(3) == "B"                       # attackers keep their colour
    assert sum(1 for v in gk.values() if v) <= 1
