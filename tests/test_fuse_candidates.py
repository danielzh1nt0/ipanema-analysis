"""ball.fuse_candidates (every app run: RF-DETR ball finder + WASB guesses -> one list per frame). N0 19 Oct."""
import numpy as np
from ipanema import ball as BL

def _get(out, k, x, y):
    return next(c for c in out[k] if abs(c[0] - x) < 1e-6 and abs(c[1] - y) < 1e-6)

def test_agreeing_guess_gets_both_scores_and_bonus_both_rules():
    rf = {0: [(100.0, 100.0, 0.4)]}; wb = {0: [(105.0, 100.0, 0.5)]}
    for nearest in (False, True):
        out = BL.fuse_candidates(rf, wb, nearest=nearest)
        assert len(out[0]) == 1
        assert abs(out[0][0][2] - (1.5 * 0.4 + 0.5 + 0.8) / 2) < 1e-9      # 0.95

def test_lone_guesses_kept_with_their_own_scale():
    out = BL.fuse_candidates({0: [(10.0, 10.0, 0.4)]}, {0: [(500.0, 500.0, 0.6)], 1: [(1.0, 1.0, 0.2)]}, nearest=True)
    assert abs(_get(out, 0, 10, 10)[2] - 0.3) < 1e-9 and abs(_get(out, 0, 500, 500)[2] - 0.3) < 1e-9
    assert set(out) == {0, 1} and abs(out[1][0][2] - 0.1) < 1e-9

def test_score_capped():
    out = BL.fuse_candidates({0: [(0.0, 0.0, 0.99)]}, {0: [(0.0, 0.0, 0.99)]})
    assert out[0][0][2] == 0.99

def test_nearest_finder_guess_takes_the_wasb_guess():
    # two finder guesses 15 px apart, the WASB guess 11 px from the first and 4 px from the SECOND: new rule boosts the second, old the first
    rf = {0: [(100.0, 100.0, 0.2), (115.0, 100.0, 0.2)]}; wb = {0: [(111.0, 100.0, 0.5)]}
    new = BL.fuse_candidates(rf, wb, nearest=True); old = BL.fuse_candidates(rf, wb, nearest=False)
    assert _get(new, 0, 115, 100)[2] > _get(new, 0, 100, 100)[2]
    assert _get(old, 0, 100, 100)[2] > _get(old, 0, 115, 100)[2]           # the old fault, pinned

def test_bonus_once_per_finder_guess():
    # two WASB peaks on one finder guess: old adds both (+ two bonuses), new takes the strongest once
    rf = {0: [(100.0, 100.0, 0.2)]}; wb = {0: [(108.0, 100.0, 0.1), (92.0, 100.0, 0.3)]}
    new = BL.fuse_candidates(rf, wb, nearest=True); old = BL.fuse_candidates(rf, wb, nearest=False)
    assert len(new[0]) == 1 and abs(new[0][0][2] - (0.3 + 0.3 + 0.8) / 2) < 1e-9
    assert abs(old[0][0][2] - min(0.99, (0.3 + 0.1 + 0.8 + 0.3 + 0.8) / 2)) < 1e-9

def test_wasb_guesses_never_merge_with_each_other():
    out = BL.fuse_candidates({}, {0: [(100.0, 100.0, 0.5), (108.0, 100.0, 0.4)]}, nearest=True)
    assert len(out[0]) == 2 and max(c[2] for c in out[0]) == 0.25

def test_default_is_the_old_rule_until_it_scores_better():
    rf = {0: [(100.0, 100.0, 0.2), (115.0, 100.0, 0.2)]}; wb = {0: [(111.0, 100.0, 0.5)]}
    assert BL.FUSE_NEAREST is False
    assert BL.fuse_candidates(rf, wb) == BL.fuse_candidates(rf, wb, nearest=False)
