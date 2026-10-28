"""N0-28b: sequences cut at the restart step's stoppages (P.split_at_stops, IPANEMA_SEQ_SPLIT_STOP) and the metres-gained
rule (P.seq_gain exact / edge<s>)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ipanema import possession as P

A, B, LOOSE = 0, 1, 2
FPS = 10


def test_split_cuts_around_a_stoppage():
    st = np.array([A] * 30 + [A] * 20 + [A] * 30)          # possession_simple keeps A through the stoppage
    seqs = [{"team": "A", "start": 0, "end": 79}]
    out = P.split_at_stops(seqs, st, [(30, 49)], min_len=5)
    assert [(s["start"], s["end"]) for s in out] == [(0, 29), (50, 79)]


def test_split_starts_after_at_the_teams_first_control_frame_and_drops_tiny_pieces():
    st = np.array([A] * 30 + [LOOSE] * 20 + [LOOSE] * 5 + [A] * 3 + [LOOSE] * 2)
    out = P.split_at_stops([{"team": "A", "start": 0, "end": 59}], st, [(30, 49)], min_len=5)
    assert [(s["start"], s["end"]) for s in out] == [(0, 29)]          # 55..57 is only 3 frames
    out = P.split_at_stops([{"team": "A", "start": 0, "end": 59}], st, [(30, 49)], min_len=1)
    assert [(s["start"], s["end"]) for s in out] == [(0, 29), (55, 57)]


def test_split_leaves_sequences_without_a_stoppage_alone_and_drops_one_inside_it():
    st = np.array([A] * 20 + [B] * 20 + [A] * 20)
    seqs = [{"team": "A", "start": 0, "end": 19}, {"team": "B", "start": 20, "end": 39}, {"team": "A", "start": 40, "end": 59}]
    out = P.split_at_stops(seqs, st, [(18, 41)], min_len=5)
    assert [(s["team"], s["start"], s["end"]) for s in out] == [("A", 0, 17), ("A", 42, 59)]
    assert P.split_at_stops(seqs, st, [], min_len=5) == seqs


def test_sequences_switch_and_end_reason():
    st = np.array([A] * 30 + [A] * 30 + [A] * 30)
    ballm = {k: (10.0 + k * 0.5, 30.0) for k in range(90)}
    kw = dict(take_s=0.5, join_s=3.0, stops=[(30, 59)], stop_s=2.0)
    old = P.sequences(st, ballm, {}, FPS, 105.0, {"A": True, "B": False}, split_stop=False, **kw)
    new = P.sequences(st, ballm, {}, FPS, 105.0, {"A": True, "B": False}, split_stop=True, **kw)
    assert len(old) == 1 and old[0]["duration_s"] == 9.0
    assert [s["duration_s"] for s in new] == [3.0, 3.0]
    assert new[0]["end_reason"] == "out of play / stoppage"
    assert new[0]["gain_m"] == 14.5 and new[1]["gain_m"] == 14.5


def test_gain_rules():
    sq = {"start": 0, "end": 19}
    ballm = {k: (50.0 + k, 30.0) for k in range(1, 18)}              # first and last frame unseen
    assert P.seq_gain(sq, ballm, 1, FPS, "exact") is None
    g = P.seq_gain(sq, ballm, 1, FPS, "edge0.5")                    # median of frames 1-4 (52.5) and 15-17 (66)
    assert abs(g - 13.5) < 1e-6
    assert abs(P.seq_gain(sq, ballm, -1, FPS, "edge0.5") + 13.5) < 1e-6
    assert P.seq_gain(sq, {}, 1, FPS, "edge0.5") is None


def test_default_is_off():
    assert P.SEQ_SPLIT_STOP is (os.environ.get("IPANEMA_SEQ_SPLIT_STOP", "0") != "0")
    assert P.SEQ_GAIN == os.environ.get("IPANEMA_SEQ_GAIN", "exact")
