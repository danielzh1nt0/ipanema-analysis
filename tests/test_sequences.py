"""E5 (29 Sep): sequences on the flickering possession_simple state. A short takeover by the other team (< take_s) must
not cut a sequence; a real change must; loose gaps up to join_s keep a team's sequence; take_s=0, join_s=1 = the old rule."""
import os, sys, random, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P
FPS = 25.0
A, B, LOOSE, DEAD = 0, 1, 2, 3


def old_spans(state, fps):
    """the rule before E5, copied from possession.sequences as it was"""
    n = len(state); seqs = []; cur = None
    for k in range(n):
        st = P.STATES[state[k]]
        if st in ("A", "B"):
            if cur and cur["team"] == st and k - cur["end"] <= int(fps): cur["end"] = k
            else:
                if cur: seqs.append(cur)
                cur = {"team": st, "start": k, "end": k}
    if cur: seqs.append(cur)
    return seqs


def st(*parts):
    return np.array([v for v, s in parts for _ in range(int(round(s * FPS)))])


def spans(state, **kw):
    return [(s["team"], s["start"], s["end"]) for s in P.sequence_spans(state, FPS, **kw)]


def test_old_rule_is_reproduced():
    rng = random.Random(1)
    for _ in range(30):
        s = np.array([rng.choice([A, A, B, B, LOOSE, DEAD]) for _ in range(rng.randint(1, 400))])
        assert spans(s, take_s=0.0, join_s=1.0) == [(q["team"], q["start"], q["end"]) for q in old_spans(s, FPS)]


def test_flicker_does_not_cut_a_sequence():
    s = st((A, 3), (B, 0.2), (A, 3))
    assert len(spans(s, take_s=0.0, join_s=1.0)) == 3                          # old rule: 3 pieces
    assert spans(s, take_s=0.5, join_s=3.0) == [("A", 0, len(s) - 1)]


def test_real_change_starts_at_the_takeover():
    s = st((A, 2), (LOOSE, 0.4), (B, 2))
    out = spans(s, take_s=0.5, join_s=3.0)
    assert [t for t, _, _ in out] == ["A", "B"]
    assert out[0][2] == int(2 * FPS) - 1 and out[1][1] == int(2.4 * FPS)


def test_flickery_takeover_still_happens():
    s = st((A, 2), (B, 0.2), (A, 0.1), (B, 0.2), (A, 0.1), (B, 2))
    out = spans(s, take_s=0.5, join_s=3.0)
    assert [t for t, _, _ in out] == ["A", "B"] and out[1][1] == int(2 * FPS)   # B's sequence starts at its first touch


def test_loose_gaps():
    assert len(spans(st((A, 2), (LOOSE, 2), (A, 2)), take_s=0.5, join_s=3.0)) == 1
    assert len(spans(st((A, 2), (LOOSE, 4), (A, 2)), take_s=0.5, join_s=3.0)) == 2
    assert [t for t, _, _ in spans(st((A, 2), (LOOSE, 4), (B, 0.2)), take_s=0.5, join_s=3.0)] == ["A", "B"]   # after a long gap anyone starts


def test_sequences_uses_new_defaults_and_keeps_fields():
    s = st((A, 3), (B, 0.2), (A, 3), (LOOSE, 1), (B, 3))
    ballm = {k: np.array([50.0 + k * 0.01, 30.0]) for k in range(len(s))}
    out = P.sequences(s, ballm, {}, FPS, 105.0, {"A": True, "B": False})
    assert [q["team"] for q in out] == ["A", "B"]
    assert {"t_start", "t_end", "duration_s", "end_reason", "gain_m", "reached_final_third"} <= set(out[0])
    assert len(P.sequences(s, ballm, {}, FPS, 105.0, {"A": True, "B": False}, take_s=0.0, join_s=1.0)) == 4


def test_metrica_truth_sequences():
    import tools.metricalab as ML
    ev = [dict(team="Home", type="PASS", sub="", period=1, f0=0, f1=20), dict(team="Home", type="CARRY", sub="", period=1, f0=20, f1=60),
          dict(team="Away", type="RECOVERY", sub="", period=1, f0=70, f1=70), dict(team="Away", type="PASS", sub="", period=1, f0=80, f1=100),
          dict(team="", type="BALL OUT", sub="", period=1, f0=110, f1=110), dict(team="Away", type="SET PIECE", sub="THROW IN", period=1, f0=200, f1=200),
          dict(team="Home", type="PASS", sub="", period=2, f0=5, f1=9)]
    t = ML.truth_sequences(ev, 1)
    assert [(q["team"], q["start"], q["end"]) for q in t] == [("A", 0, 60), ("B", 70, 100), ("B", 200, 200)]
    assert ML.seq_found(t, [(1, "A"), (75, "B"), (500, "B")]) == (2, 2)
