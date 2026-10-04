"""PC1 (4 Oct): helpers of tools/pc1lab.py (PathCRF on Metrica). Synthetic, no torch, no Metrica files."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import pc1lab as PC


def test_alive_mask_needs_ball_inside_pitch():
    bx = np.array([10.0, np.nan, -2.0, 50.0, 106.0]); by = np.array([10.0, 10.0, 30.0, 70.0, 30.0])
    assert PC.alive_mask(bx, by).tolist() == [True, False, False, False, False]


def test_episodes_join_tiny_gaps_and_drop_short_runs():
    alive = np.array([True] * 200 + [False] * 3 + [True] * 200 + [False] * 50 + [True] * 50 + [False] * 10)
    ep = PC.label_episodes(alive, np.ones(len(alive), int))
    assert set(ep[:403]) == {1}                      # 3-frame gap joined
    assert (ep[403:] == 0).all()                     # 50-frame run (< 5 s) dropped, dead frames 0


def test_episodes_break_at_period_change():
    alive = np.ones(400, bool); per = np.array([1] * 200 + [2] * 200)
    ep = PC.label_episodes(alive, per)
    assert ep[0] == 1 and ep[399] == 2


def test_pass_match_one_to_one_within_1s():
    assert PC.pass_match([100, 200, 300], [110, 105, 290, 900]) == (2, 2)   # 200 has nothing within 25 frames
    assert PC.pass_match([100], [100, 101]) == (1, 1)


def test_stats_from_edges_counts_passes_losses_and_possession():
    n = 100; idx = pd.Index(range(n), name="frame_id")
    tr = pd.DataFrame({"period_id": 1, "episode_id": 1}, index=idx)
    src = ["home_1"] * 40 + ["home_1"] * 10 + ["home_2"] * 20 + ["home_2"] * 5 + ["away_9"] * 25
    dst = ["home_1"] * 40 + ["home_2"] * 10 + ["home_2"] * 20 + ["away_9"] * 5 + ["away_9"] * 25
    micro = pd.DataFrame({"edge_src": src, "edge_dst": dst}, index=idx)
    ev = pd.DataFrame([{"frame_id": 40, "period_id": 1, "player_id": "home_1", "receiver_id": "home_2", "event_type": "kick"},
                       {"frame_id": 50, "period_id": 1, "player_id": "home_2", "receiver_id": "home_2", "event_type": "control"},
                       {"frame_id": 70, "period_id": 1, "player_id": "home_2", "receiver_id": "away_9", "event_type": "kick"},
                       {"frame_id": 75, "period_id": 1, "player_id": "away_9", "receiver_id": "away_9", "event_type": "control"}])
    st, poss = PC.stats_from_edges(tr, micro, ev, None)
    s = st[1]
    assert s["passes"] == {"Home": 1, "Away": 0} and s["balls_lost"] == {"Home": 1, "Away": 0}
    assert s["possession_pct"] == {"Home": 74, "Away": 26}            # 70 home frames (incl. flight to a teammate), 25 away, 5 in flight to opponent
    assert poss.isna().sum() == 5


def test_follow_cam_hides_far_players_and_fills_them():
    n = 250; x_far = np.r_[np.full(100, 30.0), np.linspace(30, 90, 50), np.full(100, 90.0)]
    t = pd.DataFrame({"period_id": 1, "ball_x": 30.0, "ball_y": 34.0,
                      "home_1_x": 31.0, "home_1_y": 30.0,                    # always beside the ball: seen
                      "away_2_x": x_far, "away_2_y": 40.0,                   # runs 60 m away from the ball: hidden after ~frame 110
                      "away_3_x": np.nan, "away_3_y": np.nan})               # substitute, not on the pitch
    f = PC.follow_cam(t, 35)
    assert (f["home_1_x"] == 31.0).all() and f["away_3_x"].isna().all()
    assert (f["away_2_x"].iloc[150:] == f["away_2_x"].iloc[149]).all()     # last sighting held (not his real 90 m)
    assert f["away_2_x"].iloc[-1] < 50 and 0.6 < f["seen_share"].iloc[0] < 0.8


def test_noise_std_and_smoothness():
    t = pd.DataFrame({"home_1_x": np.full(5000, 50.0), "home_1_y": np.full(5000, 30.0), "ball_x": 50.0})
    iid = PC.add_noise(t, 0.8, smooth_s=0)["home_1_x"] - 50; drift = PC.add_noise(t, 0.8, smooth_s=1.0)["home_1_x"] - 50
    assert abs(iid.std() - 0.8) < 0.05 and 0.5 < drift.std() < 1.1
    assert drift.diff().abs().mean() < 0.2 * iid.diff().abs().mean()      # drifting error moves far less frame to frame
    assert (PC.add_noise(t, 0.8)["ball_x"] == 50).all()                   # ball untouched (PathCRF never reads it)
