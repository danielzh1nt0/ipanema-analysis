"""P2c (4 Oct): the P8 per-player classifier is dropped when it moves more than 25% of one team's CLEAR colour readings
to the other team (Reymersholm 4227, Spånga 2576: it had learned something else than the kits). A classifier that agrees
with clear colour readings is kept."""
import numpy as np
from ipanema import kits as K
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_p2b_pitch_fallback import _frames

def _patch(monkeypatch, invert):
    monkeypatch.setattr(K, "on_grass", lambda f, b, g=None: True)
    monkeypatch.setattr(K, "feet_on_pitch", lambda f, b, top=None, margin=0.011: True)
    def fake_fit(H, lab, **kw):                                            # one logistic regression on the colour labels
        y = (np.asarray(lab) == "B").astype(float)
        return [K._logreg(np.asarray(H, float), 1 - y if invert else y)]
    monkeypatch.setattr(K, "fit_player_cls", fake_fit)

def test_bad_classifier_dropped(monkeypatch):
    _patch(monkeypatch, invert=True); monkeypatch.delenv("IPANEMA_CLS_MAX_FLIP", raising=False); logs = []
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=logs.append, player_cls=True)
    assert m.cls == [] and any("-> dropped, colour only" in l for l in logs)
    f, b = _frames(np.random.default_rng(1), 1)[0]; p = m.predict_batch(f, b)
    assert p[0::2] == [p[0]] * 10 and p[1::2] == [p[1]] * 10 and p[0] != p[1]   # colour decides: whites one team, reds the other

def test_good_classifier_kept(monkeypatch):
    _patch(monkeypatch, invert=False); monkeypatch.delenv("IPANEMA_CLS_MAX_FLIP", raising=False); logs = []
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=logs.append, player_cls=True)
    assert len(m.cls) == 1 and max(m.cls_flip.values()) == 0.0 and not any("dropped" in l for l in logs)

def test_selfcheck_off(monkeypatch):
    _patch(monkeypatch, invert=True); monkeypatch.setenv("IPANEMA_CLS_MAX_FLIP", "0")
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=lambda *a: None, player_cls=True)
    assert len(m.cls) == 1                                                  # old behaviour: the classifier stays
