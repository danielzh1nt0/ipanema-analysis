"""P8 (29 Sep): per-player team classifier on body colour histograms (kits.fit_player_cls)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import kits as K

def _hists(rng, n, centre, d=20):
    h = np.abs(rng.normal(0, 0.05, (n, d))); h[:, centre] += 1.0; h[:, centre + 1] += 0.5
    return h

def test_logreg_separates():
    rng = np.random.default_rng(0); X = np.r_[rng.normal(-1, 0.3, (40, 3)), rng.normal(1, 0.3, (40, 3))]; y = np.r_[np.zeros(40), np.ones(40)]
    w, b = K._logreg(X, y); assert ((X @ w + b > 0) == (y == 1)).mean() >= 0.97

def test_fixes_team_read_wrong_by_colour():
    """team B (60 people): the colour model calls 25 of them A (like the blurred floodlit whites at Reymersholm)"""
    rng = np.random.default_rng(1)
    H = np.r_[_hists(rng, 60, 2), _hists(rng, 60, 10)]; truth = np.r_[["A"] * 60, ["B"] * 60]
    lab = truth.copy(); lab[60:85] = "A"                                    # colour split 85/35: far from 50/50
    ms = K.fit_player_cls(H, lab); assert ms, "classifier should be kept (its split is nearer 50/50)"
    pred = np.array(["B" if K.player_cls_prob(ms, h) > 0.5 else "A" for h in H])
    assert (pred == truth).mean() >= 0.95 and (pred[60:85] == "B").mean() >= 0.9

def test_colour_kept_when_already_balanced():
    """if the colour split is already about even, the classifier must not take over (Spånga, SFK)"""
    rng = np.random.default_rng(2)
    H = np.r_[_hists(rng, 60, 2), _hists(rng, 60, 10)]; lab = np.r_[["A"] * 60, ["B"] * 60]
    assert K.fit_player_cls(H, lab) == []

def test_model_off_switch():
    """player_cls=False gives the plain colour model; no classifier on too few people"""
    f = np.zeros((200, 100, 3), np.uint8); f[:] = (40, 150, 40); f[40:70, 40:60] = (30, 30, 30)
    m = K.KitTeamModel().fit_frames([(f, [(30, 20, 70, 180)])] * 12, log=lambda *a: None, player_cls=False)
    assert m.cls == [] and m.predict_batch(f, [(30, 20, 70, 180)])[0] in ("A", "B", "K")
    assert K.fit_player_cls(np.zeros((5, 4)), np.array(["A"] * 5)) == []
