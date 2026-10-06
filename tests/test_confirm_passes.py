import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import analytics as AN
def test_confirm_passes():
    ps = [{"t": 1.0, "team": "A"}, {"t": 1.3, "team": "A"}, {"t": 5.0, "team": "B"}, {"t": 9.0, "team": "B"}]
    out = AN.confirm_passes(ps, [1.2, 5.5, 20.0], tol=0.7)
    assert [p["t"] for p in out] == [1.3, 5.0]
    assert AN.confirm_passes(ps, [], 0.7) == []
