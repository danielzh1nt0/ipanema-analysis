"""C3 (4 Oct): a camera row marked 'refine doubtful' is usable when forward and backward tracking agree (<= 15 px);
'cold and tracked disagree', 'no lines' still veto. 8 Oct (C3b): 'jump from track' anchors are trusted (IPANEMA_TRUST_JUMP=0 = old)."""
from ipanema.linecal import brave
P = [0.1, 0.2, 0.0, 1200.0]

def test_refine_doubtful_with_agreement_is_usable():
    assert brave({"pose": P, "fwd_bwd_px": 6.0, "why": ["refine doubtful"]})
    assert brave({"pose": P, "fwd_bwd_px": 14.9, "why": ["refine doubtful", "lines and paint disagree (12 px)"]})

def test_vetoes_stay():
    assert not brave({"pose": P, "fwd_bwd_px": 16.0, "why": ["refine doubtful"]})
    assert not brave({"pose": P, "fwd_bwd_px": 3.0, "why": ["refine doubtful", "cold and tracked disagree"]})
    assert not brave({"pose": None, "fwd_bwd_px": 3.0, "why": []})

def test_jump_from_track_trusted(monkeypatch):
    assert brave({"pose": P, "fwd_bwd_px": None, "why": ["jump from track"]})
    assert brave({"pose": P, "why": ["lines and paint disagree", "jump from track"]})
    assert not brave({"pose": P, "why": ["jump from track", "cold and tracked disagree"]})
    monkeypatch.setenv("IPANEMA_TRUST_JUMP", "0")
    assert not brave({"pose": P, "fwd_bwd_px": None, "why": ["jump from track"]})
