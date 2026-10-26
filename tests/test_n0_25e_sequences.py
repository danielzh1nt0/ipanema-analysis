"""N0-25e (26 Oct): the labels on each sequence (stats.sequences[]): 'reached_final_third' counts a sequence whose ball
was in the team's final third for at least SEQ_REACH_MIN_S (not only on its last frame), and a sequence that ends in a
loose ball running into a stoppage ends 'out of play / stoppage' (one sequence per stoppage). Old rules stay behind
IPANEMA_SEQ_REACH=end / IPANEMA_SEQ_STOP_S=0."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import possession as P
FPS = 25.0; L, W = 105.0, 68.0
A, B, LOOSE, DEAD = 0, 1, 2, 3
AR = {"A": True, "B": False}


def st(*parts):
    return np.array([v for v, s in parts for _ in range(int(round(s * FPS)))])


def ball_path(n, xs):
    """xs: list of (frames, x) pieces -> {frame: [x, 30]}"""
    out = {}; k = 0
    for f, x in xs:
        for _ in range(f): out[k] = np.array([float(x), 30.0]); k += 1
    while k < n: out[k] = np.array([50.0, 30.0]); k += 1
    return out


def test_reach_any_frame_vs_last_frame():
    s = st((A, 4))                                     # A attacks right: final third = x > 70
    bm = ball_path(len(s), [(25, 50), (25, 80), (50, 55)])   # 1 s in the final third, then pushed back
    old = P.sequences(s, bm, {}, FPS, L, AR, reach="end", stop_s=0)
    new = P.sequences(s, bm, {}, FPS, L, AR, reach="any", reach_min_s=0.5, stop_s=0)
    assert old[0]["reached_final_third"] is False and new[0]["reached_final_third"] is True


def test_reach_needs_min_time():
    s = st((A, 4))
    bm = ball_path(len(s), [(40, 50), (5, 80), (55, 55)])    # 0.2 s in the final third: a picker jump
    assert P.sequences(s, bm, {}, FPS, L, AR, reach="any", reach_min_s=0.0, stop_s=0)[0]["reached_final_third"] is True
    assert P.sequences(s, bm, {}, FPS, L, AR, reach="any", reach_min_s=0.5, stop_s=0)[0]["reached_final_third"] is False
    bm2 = ball_path(len(s), [(95, 50), (5, 80)])            # ends in it: true under every rule (old rule kept)
    assert P.sequences(s, bm2, {}, FPS, L, AR, reach="any", reach_min_s=0.5, stop_s=0)[0]["reached_final_third"] is True


def test_reach_side_follows_direction():
    s = st((B, 4))                                     # B attacks left: final third = x < 35
    bm = ball_path(len(s), [(25, 50), (25, 20), (50, 55)])
    assert P.sequences(s, bm, {}, FPS, L, AR, reach="any", reach_min_s=0.5, stop_s=0)[0]["reached_final_third"] is True
    bm = ball_path(len(s), [(25, 50), (25, 90), (50, 55)])   # in A's final third, not B's
    assert P.sequences(s, bm, {}, FPS, L, AR, reach="any", reach_min_s=0.5, stop_s=0)[0]["reached_final_third"] is False


def test_stoppage_ends_a_loose_ball_sequence_once():
    s = st((A, 3), (LOOSE, 1), (LOOSE, 6), (A, 3), (LOOSE, 4), (B, 3))
    bm = ball_path(len(s), [])
    stops = [(int(4 * FPS), int(10 * FPS) - 1)]        # play stopped 1 s after A's last touch
    old = P.sequences(s, bm, {}, FPS, L, AR, stop_s=0)
    new = P.sequences(s, bm, {}, FPS, L, AR, stops=stops, stop_s=2.0)
    assert [q["end_reason"] for q in old][:2] == ["loose ball", "loose ball"]
    assert new[0]["end_reason"] == "out of play / stoppage"
    assert new[1]["end_reason"] == "loose ball"        # the next sequence does not take the same stoppage
    late = [(int(6 * FPS), int(10 * FPS) - 1)]          # the stoppage starts 3 s after the end: not this sequence's
    assert P.sequences(s, bm, {}, FPS, L, AR, stops=late, stop_s=2.0)[0]["end_reason"] == "loose ball"


def test_interception_is_not_relabelled():
    s = st((A, 3), (B, 3))
    stops = [(int(3 * FPS), int(6 * FPS) - 1)]
    out = P.sequences(s, ball_path(len(s), []), {}, FPS, L, AR, stops=stops, stop_s=2.0)
    assert out[0]["end_reason"] in ("pass intercepted", "dispossessed")


def test_stoppage_spans_match_restarts():
    s = st((A, 3), (DEAD, 4), (A, 5), (DEAD, 1), (LOOSE, 1), (DEAD, 2), (B, 5), (DEAD, 1), (A, 2))   # 2nd joined across 1 s; last too short
    bm = {k: np.array([50.0, 30.0]) for k in range(len(s))}
    sp = P.stoppage_spans(s, bm, FPS, L, W, gap_s=0)
    rs = P.restarts(s, bm, FPS, L, W, gap_s=0)
    assert [round(j / FPS, 2) for _, j in sp] == [r["t"] for r in rs] and len(sp) == 2


def test_defaults_and_pipeline_wiring():
    assert P.SEQ_REACH == "any" and P.SEQ_STOP_S == 2.0 and P.SEQ_REACH_MIN_S == 0.5
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipanema", "run.py")).read()
    assert "stops=P.stoppage_spans(dstate" in src
