"""N0-24e (24 Oct): an unseen-ball gap next to a line counts as a stoppage (possession.gap_dead / restarts(gap_s=...)),
the restart spot is where the ball comes back, the team is the player nearest to it, goal kicks from the 6-m box.
Plus the Metrica truth reader of tools/n0_24e_restarts.py."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from ipanema import possession as P

FPS = 10.0; L, W = 105.0, 68.0
A, B, LOOSE, DEAD = 0, 1, 2, 3


def _throw_in(gap_frames=40, last=(50.0, 67.0), back=(52.0, 68.3)):
    """team A has the ball, it rolls out over the far touchline, nobody sees it for gap_frames, B throws it back in"""
    n = 30 + gap_frames + 30; st = [A] * 30 + [A] * gap_frames + [B] * 30
    ballm = {k: np.array([40.0 + k / 3, 60.0 + k / 5], float) for k in range(30)}; ballm[29] = np.array(last, float)
    for k in range(30 + gap_frames, n): ballm[k] = np.array(back, float) + (k - 30 - gap_frames) * np.array([0.0, -0.5])
    per = {k: [[1, "A", np.array([30.0, 30.0])], [2, "B", np.array(back) + np.array([0.0, 0.5])]] for k in range(n)}
    return np.asarray(st), ballm, per


def test_old_rule_never_sees_an_unseen_stoppage():
    st, ballm, per = _throw_in()
    assert P.restarts(st, ballm, FPS, L, W, gap_s=0) == []


def test_gap_next_to_a_line_is_a_restart_with_spot_kind_and_team():
    st, ballm, per = _throw_in()
    r = P.restarts(st, ballm, FPS, L, W, gap_s=2.0, per=per)
    assert len(r) == 1
    assert r[0]["t"] == 6.9 and r[0]["kind"] == "throw-in" and r[0]["team"] == "B"
    assert abs(r[0]["x_m"] - 52.0) < 0.5 and r[0]["y_m"] > 67.0             # where it came back, not 1.5 s later
    r2 = P.restarts(st, ballm, FPS, L, W, gap_s=2.0)                         # no players given: the old team rule (state after)
    assert r2[0]["team"] == "B"


def test_short_gap_or_gap_in_midfield_is_not_a_stoppage():
    st, ballm, per = _throw_in(gap_frames=15)                                # 1.5 s < 2 s
    assert P.restarts(st, ballm, FPS, L, W, gap_s=2.0, per=per) == []
    st, ballm, per = _throw_in(last=(50.0, 30.0), back=(55.0, 34.0))         # ball lost in the middle (e.g. in the air)
    assert P.restarts(st, ballm, FPS, L, W, gap_s=2.0, per=per) == []


def test_gap_dead_sides():
    st, ballm, per = _throw_in(last=(50.0, 30.0))                           # only the reappearance is at the line
    n = len(st)
    assert (P.gap_dead(st, ballm, n, FPS, L, W, 2.0, 3.0, "before") == 3).sum() == 0
    assert (P.gap_dead(st, ballm, n, FPS, L, W, 2.0, 3.0, "after") == 3).sum() == 40
    assert (P.gap_dead(st, ballm, n, FPS, L, W, 2.0, 3.0, "either") == 3).sum() == 40
    assert (P.gap_dead(st, {}, n, FPS, L, W, 2.0) == 3).sum() == 0          # never saw the ball at all: no evidence


def test_goal_kick_from_the_six_metre_box_only_with_the_gap_rule():
    st = [A] * 10 + [DEAD] * 40 + [B] * 40; ballm = {k: np.array([4.8, 33.0]) for k in range(10, 90)}
    assert P.restarts(np.asarray(st), ballm, FPS, L, W, gap_s=0)[0]["kind"] == "stoppage"      # old: within 3 m of the line only
    assert P.restarts(np.asarray(st), ballm, FPS, L, W, gap_s=2.0)[0]["kind"] == "goal kick"


def test_default_is_on_and_env_switch_exists():
    assert P.GAP_DEAD_S == float(os.environ.get("IPANEMA_GAP_DEAD_S", "2"))
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ipanema", "run.py")).read()
    assert "P.restarts(dstate, ballm, fps, L, W, per=per)" in src


def test_metrica_truth_reader():
    import n0_24e_restarts as R
    rows = [{"Period": "1", "Team": "Home", "Type": "SET PIECE", "Subtype": "THROW IN", "Start Frame": "100"},
            {"Period": "1", "Team": "Away", "Type": "SET PIECE", "Subtype": "GOAL KICK", "Start Frame": "500"},
            {"Period": "1", "Team": "Away", "Type": "PASS", "Subtype": "GOAL KICK", "Start Frame": "500"},
            {"Period": "1", "Team": "Away", "Type": "PASS", "Subtype": "", "Start Frame": "700"},
            {"Period": "2", "Team": "Home", "Type": "SET PIECE", "Subtype": "CORNER KICK", "Start Frame": "900"}]
    t = R.truth_restarts(rows, 1)
    assert [(x["frame"], x["kind"], x["team"]) for x in t] == [(100, "throw-in", "A"), (500, "goal kick", "B")]
    ours = [{"frame": 110, "kind": "throw-in", "team": "A"}, {"frame": 2000, "kind": "stoppage", "team": None}]
    s = R.score(t, ours, 10.0)
    assert s["found"] == 1 and s["extra"] == 1 and s["kind_right_out_of_play"] == 1 and s["team_right"] == 1
