"""S8 (2 Oct): the counted stats read a de-flickered possession state; the raw state is untouched."""
import numpy as np
from ipanema import possession as P

def test_short_flickers_are_absorbed():
    fps = 30.0; st = np.zeros(300, int)                       # A has it for 10 s ...
    st[100:110] = 1; st[150:160] = 1                           # ... with two 0.33-s flickers to B
    out = P.spell_state(st, fps, 1.5, 3.0)
    assert (out == 0).all()

def test_real_takeover_switches():
    fps = 30.0; st = np.zeros(300, int); st[150:] = 1          # B takes it for the last 5 s
    out = P.spell_state(st, fps, 1.5, 3.0)
    assert (out[:150] == 0).all() and (out[150:] == 1).all()

def test_off_returns_raw():
    st = np.array([0, 1, 0, 1, 2, 3]); assert (P.spell_state(st, 30.0, 0, 3.0) == st).all()

def test_loose_outside_spans_kept():
    fps = 30.0; st = np.full(300, 2, int); st[:60] = 0          # 2 s of A, then 8 s loose
    out = P.spell_state(st, fps, 1.5, 3.0)
    assert (out[:60] == 0).all() and (out[200:] == 2).all()
