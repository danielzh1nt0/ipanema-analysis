"""4 Oct (Vallentuna, red vs black in hard sun): a SMALL bright group (sun-washed players, 10 of 57) must not become team B
under the far-pair rule just because it is least like the black team; the red team (as big as the blacks) must."""
import numpy as np
from ipanema import kits as K

def test_small_bright_group_is_not_a_team():
    rng = np.random.default_rng(1)
    black, red, bright = np.array([18.0, 5, -17]), np.array([35.0, 31, -10]), np.array([70.0, 1, 10])
    X = np.vstack([c + rng.normal(0, 1.5, (n, 3)) for c, n in ((black, 42), (red, 40), (bright, 12))])
    m = K.fit(list(X), pair="far"); tb = max(m["teams"], key=lambda c: c[1])
    assert np.linalg.norm(tb - red) < 5, m["teams"]
