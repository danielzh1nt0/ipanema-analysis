"""2 Oct: pressures_applied counts one per presser per possession sequence when sequences are given (not one per second)."""
import numpy as np
from ipanema import analytics as AN

def _setup(n=90, fps=30.0):
    # carrier 1 (A) stands at (10,10) for 3 s; opponent 2 (B) stays within 2 m the whole time; teammate 3 (A) far away
    per = {k: [[1, "A", np.array([10.0, 10.0]), None, None, False], [2, "B", np.array([11.0, 10.0]), None, None, False], [3, "A", np.array([40.0, 30.0]), None, None, False]] for k in range(n)}
    frames_ = [{"carrier": 1, "team": "A", "pressure_m": 1.0, "near_opps": 1, "pos": np.array([10.0, 10.0])} for _ in range(n)]
    tracks = {1: {k: per[k][0][2] for k in range(n)}, 2: {k: per[k][1][2] for k in range(n)}, 3: {k: per[k][2][2] for k in range(n)}}
    state = np.zeros(n, int)
    return per, frames_, tracks, state, fps

def _pressures(st, tid):
    return next(p["pressures_applied"] for p in st["players"] if p["id"] == tid)

def test_per_second_without_sequences():
    per, frames_, tracks, state, fps = _setup()
    st = AN.stats(per, frames_, [], [], tracks, state, fps, 105.0, 68.0, {"A": True, "B": False})
    assert _pressures(st, 2) == 3          # one per second over 3 s (old behaviour)

def test_once_per_sequence():
    per, frames_, tracks, state, fps = _setup()
    seqs = [{"team": "A", "start": 0, "end": 89}]
    st = AN.stats(per, frames_, [], [], tracks, state, fps, 105.0, 68.0, {"A": True, "B": False}, sequences_=seqs)
    assert _pressures(st, 2) == 1
    assert _pressures(st, 3) == 0          # teammates never press
    seqs = [{"team": "A", "start": 0, "end": 44}, {"team": "A", "start": 45, "end": 89}]
    st = AN.stats(per, frames_, [], [], tracks, state, fps, 105.0, 68.0, {"A": True, "B": False}, sequences_=seqs)
    assert _pressures(st, 2) == 2          # a new sequence = a new pressure
    assert next(t for t in st["teams"] if t["team"] == "B")["pressures_applied"] == 2
