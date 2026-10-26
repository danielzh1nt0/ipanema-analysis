"""N0-26d (26 Oct): per-player time on the ball / under pressure (stats.json players[].time_on_ball_s,
time_under_pressure_s) count only the carrier frames inside his cleaned contacts (analytics.touch_frames), so the
player table can't show seconds on the ball with 0 touches; IPANEMA_ONBALL=old = every carrier frame."""
import os, sys, numpy as np
from ipanema import analytics as AN

FPS = 30.0
SEQ = [(1, "A", 30), (9, "B", 15), (1, "A", 15), (None, None, 10), (2, "A", 30), (9, "B", 1), (None, None, 20)]


def _setup(seq, pressure=None):
    frames_ = []
    for c, tm, n in seq:
        frames_ += [{"carrier": c, "team": tm if c is not None else None, "pressure_m": pressure, "near_opps": 0, "pos": None} for _ in range(n)]
    n = len(frames_); spots = {1: (10.0, 10.0), 2: (30.0, 10.0), 9: (31.0, 11.0)}
    per = {k: [[t, "A" if t < 9 else "B", np.array(xy), None, None, False] for t, xy in spots.items()] for k in range(n)}
    tracks = {t: {k: np.array(xy) for k in range(n)} for t, xy in spots.items()}
    return per, frames_, tracks, np.zeros(n, int)


def _p(st, tid):
    return next(p for p in st["players"] if p["id"] == tid)


def test_touch_time_only_inside_contacts():
    per, frames_, tracks, _ = _setup(SEQ)
    eps = AN.touch_episodes(frames_, FPS)
    t = AN.touch_time(frames_, eps, FPS)
    assert abs(t[1] - 45 / FPS) < 1e-9          # 1's 30 + 15 frames; the flicker frames are 9's, not 1's
    assert abs(t[2] - 1.0) < 1e-9 and 9 not in t  # 9's flicker and blip are no contact of his


def test_stats_time_on_ball_new_and_old_switch(monkeypatch):
    per, frames_, tracks, state = _setup(SEQ, pressure=1.0)
    eps = AN.touch_episodes(frames_, FPS); ar = {"A": True, "B": False}
    old = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, ar)
    assert _p(old, 9)["time_on_ball_s"] == round(16 / FPS, 1)
    new = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, ar, episodes_=eps)
    assert _p(new, 9)["time_on_ball_s"] == 0.0 and _p(new, 9)["touches"] == 0 and _p(new, 9)["time_under_pressure_s"] == 0.0
    assert _p(new, 1)["time_on_ball_s"] == 1.5 and _p(new, 1)["time_under_pressure_s"] == 1.5
    monkeypatch.setenv("IPANEMA_ONBALL", "old")
    sw = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, ar, episodes_=eps)
    assert [p["time_on_ball_s"] for p in sw["players"]] == [p["time_on_ball_s"] for p in old["players"]]


def test_no_time_without_touch():
    per, frames_, tracks, state = _setup(SEQ)
    eps = AN.touch_episodes(frames_, FPS)
    st = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, {"A": True, "B": False}, episodes_=eps)
    for p in st["players"]:
        assert (p["time_on_ball_s"] > 0) == (p["touches"] > 0)


def test_lab_truth_spells(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import n0_26d_onball as L
    f = tmp_path / "ev.csv"
    f.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
                 "Away,SET PIECE,KICK OFF,1,1,0,0,0,Player19,,,,,\n"
                 "Away,PASS,,1,1,0,3,0,Player19,Player21,,,,\n"      # one-touch from the kick-off spot: 0 s
                 "Away,PASS,,1,53,0,60,0,Player21,Player15,,,,\n"    # 21 had it from frame 3 to 53 = 2 s
                 "Away,BALL LOST,,1,85,0,90,0,Player15,,,,,\n"       # 15 from 60 to 85 = 1 s
                 "Home,RECOVERY,,1,90,0,90,0,Player2,,,,,\n"
                 "Home,SHOT,,1,115,0,120,0,Player2,,,,,\n")          # 2 from 90 to 115 = 1 s
    tob, rec, tch = L.truth(str(f), 1)
    assert tob == {219: 0.0, 221: 2.0, 215: 1.0, 102: 1.0}
    assert rec == {221: 1, 215: 1} and tch == {219: 1, 221: 1, 215: 1, 102: 1}
