"""P1 (29 Sep): striped kits. A striped shirt's MEDIAN colour jumps between its stripes, so many striped players read as
'neither team' (Spånga: 79/142). The kit fit now also tries a mean colour and the far-pair rule and keeps the reading that
leaves the fewest people as 'neither' (only if it wins by 3 points)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

def person(rng, kind):
    im = np.zeros((120, 30, 3), np.uint8); im[:] = (40, 150, 40)
    if kind == "dark": im[14:40, 6:24] = np.clip(np.array((30, 30, 35)) + rng.normal(0, 8, 3), 0, 255)
    elif kind == "ref": im[14:40, 6:24] = np.clip(np.array((20, 190, 235)) + rng.normal(0, 8, 3), 0, 255)
    else:                                                        # white/navy vertical stripes, share of white varies with pose
        share = rng.uniform(0.3, 0.7); w = 6; x = np.arange(6, 24); white = ((x - 6 + rng.integers(0, w)) % w) < share * w
        for i, xx in enumerate(x): im[14:40, xx] = np.clip(np.array((200, 225, 235) if white[i] else (90, 35, 25)) + rng.normal(0, 6, 3), 0, 255)
    return im

def fit_and_read(auto):
    rng = np.random.default_rng(1); os.environ["IPANEMA_KIT_AUTO"] = "1" if auto else "0"
    try:
        kinds = ["dark"] * 60 + ["striped"] * 60 + ["ref"] * 5; ims = [person(rng, k) for k in kinds]
        tm = K.KitTeamModel().fit_frames([(im, [(0, 0, 30, 120)]) for im in ims], log=lambda *a: None, player_cls=False)
        labs = [tm.predict_batch(im, [(0, 0, 30, 120)])[0] for im in ims]
        return tm, labs[:60], labs[60:120], labs[120:]
    finally: os.environ.pop("IPANEMA_KIT_AUTO", None)

def test_mean_reading_is_steadier_on_stripes():
    rng = np.random.default_rng(2); ims = [person(rng, "striped") for _ in range(40)]
    sd = {s: np.std([K.torso_feature(im, (0, 0, 30, 120), stat=s)[0] for im in ims]) for s in ("median", "mean")}
    assert sd["mean"] < 0.6 * sd["median"], sd

def test_striped_team_read_as_a_team():
    tm, dark, striped, ref = fit_and_read(True)
    assert dark.count("A") >= 57 and striped.count("B") >= 54, (dark.count("A"), striped.count("B"), tm.choice)
    assert tm.stat == "mean" and len(tm.choice) == 4

def test_auto_off_is_the_old_reading():
    tm, dark, striped, ref = fit_and_read(False)
    assert tm.stat is None and len(tm.choice) == 1
