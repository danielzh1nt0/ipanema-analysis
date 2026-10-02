"""F1b (2 Oct): per-piece kit models named like the match model (no A/B swaps between pieces)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

rng = np.random.default_rng(1)
def person(bgr, light=0):
    im = np.zeros((120, 30, 3), np.uint8); im[:] = (40, 150, 40)
    im[14:40, 6:24] = np.clip(np.array(bgr) + light + rng.normal(0, 6, 3), 0, 255); return im
def frames(light, n=(30, 30, 3)):
    ppl = [person((150, 40, 40), light) for _ in range(n[0])] + [person((40, 200, 230), light) for _ in range(n[1])] + [person((30, 30, 30), light) for _ in range(n[2])]
    return [(im, [(0, 0, 30, 120)]) for im in ppl]
Q = lambda *a: None
def labels(m, fb): return [m.predict_batch(f, b)[0] for f, b in fb]

def test_piece_model_named_like_match_and_reads_its_own_light():
    match = K.KitTeamModel().fit_frames(frames(0), log=Q, player_cls=False)
    lm = labels(match, frames(0)); blue, yellow = max("AB", key=lm[:30].count), max("AB", key=lm[30:60].count)
    assert blue != yellow
    sunny = frames(70)                                     # strong sun: the match colours no longer fit
    pm, info = K.piece_model(sunny, match, log=Q)
    assert info["fallback"] is None and pm is not match
    lp = labels(pm, sunny)
    assert lp[:30].count(blue) >= 28 and lp[30:60].count(yellow) >= 28      # same team names as the match model
    assert lp[:60].count("K") <= labels(match, sunny)[:60].count("K")       # no more 'neither' than the match model

def test_piece_model_flips_when_its_names_are_swapped():
    match = K.KitTeamModel().fit_frames(frames(0), log=Q, player_cls=False)
    swapped = K.KitTeamModel().fit_frames(frames(0), log=Q, player_cls=False); swapped.flip = True   # a match model with reversed names
    pm, info = K.piece_model(frames(10), swapped, log=Q)
    assert info["flip"] and info["fallback"] is None
    lp, ls, lm = labels(pm, frames(0))[:60], labels(swapped, frames(0))[:60], labels(match, frames(0))[:60]
    assert sum(a == b for a, b in zip(lp, ls)) >= 56 and sum(a != b for a, b in zip(lp, lm)) >= 56

def test_piece_model_falls_back_with_few_people():
    match = K.KitTeamModel().fit_frames(frames(0), log=Q, player_cls=False)
    pm, info = K.piece_model(frames(0, n=(5, 5, 1)), match, log=Q)
    assert pm is match and info["fallback"]

def test_old_pickles_without_flip_still_read():
    m = K.KitTeamModel().fit_frames(frames(0), log=Q, player_cls=False); del m.__dict__["strips"]
    assert not hasattr(m, "flip") or m.flip is False
    assert set(labels(m, frames(0))) <= {"A", "B", "K"}
