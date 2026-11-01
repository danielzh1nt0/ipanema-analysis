"""N0-01b (1 Nov): the app numbers agree with each other (tools/n0_01b_consistency.py) and the blind pass sheets
(tools/n0_01b_moments.py / n0_01b_frames.py) are built right."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_01b_consistency as C
import n0_01b_frames as F
import n0_01b_moments as M


def _stats():
    pl = [{"id": 1, "team": "A", "passes": 2, "passes_completed": 1, "passes_forward": 1, "passes_sideways": 1, "passes_backward": 0, "progressive_passes": 1,
           "passes_risky": 0, "pass_quality": {"good": 1}, "pct_def_third": 30, "pct_mid_third": 40, "pct_att_third": 30, "time_visible_s": 10, "time_on_ball_s": 2,
           "time_under_pressure_s": 1, "touches": 2, "distance_m": 50, "high_intensity_m": 5, "top_speed_ms": 6, "pressures_applied": 1, "better_option_count": 0},
          {"id": 2, "team": "B", "passes": 1, "passes_completed": 1, "passes_forward": 0, "passes_sideways": 0, "passes_backward": 1, "progressive_passes": 0,
           "passes_risky": 0, "pass_quality": {}, "pct_def_third": 100, "pct_mid_third": 0, "pct_att_third": 0, "time_visible_s": 10, "time_on_ball_s": 1,
           "time_under_pressure_s": 0, "touches": 1, "distance_m": 40, "high_intensity_m": 0, "top_speed_ms": 5, "pressures_applied": 2, "better_option_count": 1}]
    ps = [{"team": "A", "from": 1, "completed": True, "kind": "forward", "progressive": True, "better_option": False},
          {"team": "A", "from": 1, "completed": False, "kind": "sideways", "progressive": False, "better_option": False},
          {"team": "B", "from": 2, "completed": True, "kind": "backward", "progressive": False, "better_option": True}]
    tm = [{"team": "A", "passes": 2, "pass_completion_pct": 50, "forward_pass_share_pct": 50, "progressive_passes": 1, "sequences": 1, "passes_per_sequence": 2.0,
           "distance_m_total_visible": 50, "pressures_applied": 1, "better_option_count": 0, "possession_pct": 60, "possession_s": 6, "losses": 1, "recoveries": 2},
          {"team": "B", "passes": 1, "pass_completion_pct": 100, "forward_pass_share_pct": 0, "progressive_passes": 0, "sequences": 1, "passes_per_sequence": 1.0,
           "distance_m_total_visible": 40, "pressures_applied": 2, "better_option_count": 1, "possession_pct": 40, "possession_s": 4, "losses": 2, "recoveries": 1}]
    return {"players": pl, "passes": ps, "teams": tm, "sequences": [{"team": "A"}, {"team": "B"}]}


def test_clean_stats_pass():
    assert C.check(_stats()) == []


def test_broken_rules_found():
    s = _stats(); s["teams"][0]["losses"] = 5; s["players"][0]["passes_forward"] = 2; s["teams"][1]["possession_pct"] = 30
    s["players"][1]["team"] = "A"                      # B's passer listed in team A -> A's players pass more than A
    names = {b["check"] for b in C.check(s)}
    assert "balls lost by one = balls won by the other" in names
    assert "forward + sideways + backward = passes" in names
    assert "possession A + B = 100%" in names
    assert "players' passes <= team passes" in names


def test_repo_copies_mostly_agree():
    """on the repo's app copies only the known faults remain (new faults would show here)"""
    for p in glob.glob("results/volume/runs/matches/*/stats.json"):
        s = json.load(open(p)); b = C.check(s)
        assert {x["check"] for x in b} <= {"team better-option = sum of players", "players' passes <= team passes", "team sequences = sequences list"}, (p, b[:3])


def test_crop_keeps_player_inside():
    for px in ([5, 5], [1915, 1075], [960, 540]):
        x0, y0 = F.crop_box(px); assert 0 <= x0 <= 1920 - F.CW and 0 <= y0 <= 1080 - F.CH
        assert x0 <= px[0] <= x0 + F.CW and y0 <= px[1] <= y0 + F.CH


def test_sheet_shape():
    import cv2 as cv
    f = np.zeros((1080, 1920, 3), np.uint8)
    im = F.sheet(cv, np, f, {"id": "p00", "passer_px": [100, 900], "ref1_px": [1900, 10], "ref2_px": [960, 540]})
    assert im.shape[1] == 3 * 2 * F.CW and im.shape[0] == 2 * F.CH + int(im.shape[1] * 1080 / 1920)


def test_refs_pick_nearest_steady_player():
    import collections
    f = {"players": [{"id": 1, "team": "A", "px": [0, 0], "state": "observed"}, {"id": 2, "team": "B", "px": [10, 0], "state": "observed"},
                     {"id": 3, "team": "B", "px": [5, 0], "state": "observed"}, {"id": 4, "team": "B", "px": [50, 0], "state": "observed"}]}
    tt = {1: collections.Counter(A=3), 2: collections.Counter(B=3), 3: collections.Counter(A=1, B=2), 4: collections.Counter(B=1)}
    assert [q["id"] for q in M.refs(f, 1, tt, "B")] == [2]           # 3 is nearer but its track changes team


def test_blind_file_has_no_answers():
    p = "results/qa/n0-01b/moments.json"
    if not os.path.exists(p): return
    for m in json.load(open(p)):
        assert set(m) == {"id", "match", "t", "passer_px", "ref1_px", "ref2_px"}
