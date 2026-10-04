"""P2b (4 Oct): when the grass test keeps under 35% of the people (Reymersholm at night: 12-29%), the kit model is learned
from the people the pitch-edge test keeps instead, so the second team is not left out of the kit groups."""
import numpy as np
from ipanema import kits as K

def _frames(rng, n=6):
    out = []
    for _ in range(n):
        f = np.zeros((1080, 1920, 3), np.uint8); f[:] = (40, 150, 40); boxes = []
        for i in range(20):
            x, y = 60 + 90 * i, 400 + 20 * (i % 5); col = (30, 30, 200) if i % 2 else (235, 235, 235)
            f[y:y + 80, x:x + 30] = np.clip(np.array(col) + rng.normal(0, 4, 3), 0, 255).astype(np.uint8)
            boxes.append([x, y, x + 30, y + 80])
        out.append((f, np.array(boxes, float)))
    return out

def _patch(monkeypatch, grass_keeps):
    # grass test keeps only the first people of the white team (like the floodlit whites dropped at night); edge keeps all
    monkeypatch.setattr(K, "on_grass", lambda f, b, g=None: b[0] < 60 + 90 * grass_keeps)
    monkeypatch.setattr(K, "feet_on_pitch", lambda f, b, top=None, margin=0.011: True)

def test_fallback_to_edge_when_grass_keeps_few(monkeypatch):
    _patch(monkeypatch, 4); monkeypatch.delenv("IPANEMA_PITCH_FALLBACK", raising=False); logs = []
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=logs.append, player_cls=False)
    assert m.pitch_test == "edge (fallback)" and any("pitch-edge test instead" in l for l in logs)
    assert len(m.samples) == 120                                              # everyone on the pitch, both kits
    f, b = _frames(np.random.default_rng(1), 1)[0]; p = m.predict_batch(f, b)
    assert p.count("A") >= 9 and p.count("B") >= 9                            # red and white both read as teams

def test_no_fallback_when_grass_keeps_most(monkeypatch):
    _patch(monkeypatch, 14); monkeypatch.delenv("IPANEMA_PITCH_FALLBACK", raising=False)
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=lambda *a: None, player_cls=False)
    assert m.pitch_test == "grass" and len(m.samples) == 6 * 14

def test_fallback_off(monkeypatch):
    _patch(monkeypatch, 4); monkeypatch.setenv("IPANEMA_PITCH_FALLBACK", "0")
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=lambda *a: None, player_cls=False)
    assert m.pitch_test == "grass" and len(m.samples) == 6 * 4

def test_no_fallback_in_hard_sun(monkeypatch):
    """4 Oct, Vallentuna: the grass test keeps 20% there too, but the light is uneven (sun + shade) and the edge test let the
    bench in and the fit became sun vs shade. Uneven light -> grass test kept."""
    _patch(monkeypatch, 4); monkeypatch.delenv("IPANEMA_PITCH_FALLBACK", raising=False); logs = []
    monkeypatch.setattr(K, "shade_spread", lambda kept: 2.6)
    m = K.KitTeamModel().fit_frames(_frames(np.random.default_rng(0)), log=logs.append, player_cls=False)
    assert m.pitch_test == "grass" and any("light is uneven" in l for l in logs)
