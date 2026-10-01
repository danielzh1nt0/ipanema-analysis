"""P1 v3 (1 Oct): the striped-kit switch must fire on the REAL tracking pieces where the default picks the wrong pair of
teams (Spånga striped kits; Djursholm 2714, where spectators became a team and the red team read 'neither'), and must
not fire on plain-kit pieces (there far pair + mean colour was mixed by eye: Solberga referees joined a team).
Pieces = the kit step's own input saved on Kaggle (results/kaggle/kitcrops_p1, tools/p1crops.py)."""
import os, sys, numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import kits as K
import p1crops as P
SRC = f"{ROOT}/results/kaggle/kitcrops_p1"

def fit(tag):
    old = os.environ.get("IPANEMA_KIT_CLS"); os.environ["IPANEMA_KIT_CLS"] = "0"
    try: return P.fit(P.load(f"{SRC}/{tag}.pkl.gz"))
    finally:
        if old is None: os.environ.pop("IPANEMA_KIT_CLS", None)
        else: os.environ["IPANEMA_KIT_CLS"] = old

@pytest.mark.parametrize("tag", ["p15u-vs-spanga-2026-09-25_1125", "p15u-vs-djursholm-2026-09-26_2714"])
def test_switch_fires_where_default_picks_wrong_teams(tag):
    tm, lines, rd, flat, fb = fit(tag)
    assert tm.pair == "far" and tm.stat == "mean" and tm.pair_shift > 18, lines[-1]
    if "spanga" in tag:                                # near striped players were 'neither': 99 of 633 on the pitch
        on = [r for r in rd if r != "O"]; assert on.count("K") < 0.05 * len(on), lines[-1]

@pytest.mark.parametrize("tag", ["SFKBP1109_s1200_20", "solberga-vs-p09-norrviken-2026-09-11_2766", "p15u-vs-reymersholm-2026-09-18_4286", "p15u-vs-vasalund-2026-09-20_4068"])
def test_switch_stays_off_on_plain_kits(tag):
    tm, lines, rd, flat, fb = fit(tag)
    assert tm.stat is None and len(tm.choice) == 1 and tm.pair_shift <= 18, lines[-1]

def test_far_pick_prefers_colour_over_darkness():
    """three big groups: white (team A), red (the other kit), dark coats (spectators, darker than red). The second team
    must be red: with plain distance the dark coats won (Djursholm 1428/2714)."""
    rng = np.random.default_rng(0)
    white, red, dark = np.array([58.0, -2, 2]), np.array([32.0, 26, 22]), np.array([14.0, -3, 4])
    X = np.vstack([c + rng.normal(0, 1.5, (n, 3)) for c, n in ((white, 220), (red, 195), (dark, 185))])
    m = K.fit(list(X), pair="far"); tb = min(m["teams"], key=lambda c: c[0])
    assert np.linalg.norm(tb - red) < 5, m["teams"]
    old = os.environ.get("IPANEMA_FAR_PICK"); os.environ["IPANEMA_FAR_PICK"] = "plain"
    try: mp = K.fit(list(X), pair="far")
    finally:
        if old is None: os.environ.pop("IPANEMA_FAR_PICK", None)
        else: os.environ["IPANEMA_FAR_PICK"] = old
    assert np.linalg.norm(min(mp["teams"], key=lambda c: c[0]) - dark) < 5          # the old rule, for the record
