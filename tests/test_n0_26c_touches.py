"""N0-26c (26 Oct): per-player touches (stats.json players[].touches, in the app's player table) come from the same
cleaned ball contacts the passes are built from (analytics.touch_episodes), not from every run of carrier frames."""
import os, sys, importlib, numpy as np
from ipanema import analytics as AN

FPS = 30.0


def _frames(seq):
    """seq: list of (player or None, team, n frames) -> frames_ records like possession.carriers"""
    out = []
    for c, tm, n in seq:
        out += [{"carrier": c, "team": tm if c is not None else None, "pressure_m": None, "near_opps": 0, "pos": None} for _ in range(n)]
    return out


def _setup(seq):
    frames_ = _frames(seq); n = len(frames_)
    spots = {1: (10.0, 10.0), 2: (30.0, 10.0), 9: (31.0, 11.0)}
    per = {k: [[t, "A" if t < 9 else "B", np.array(xy), None, None, False] for t, xy in spots.items()] for k in range(n)}
    tracks = {t: {k: np.array(xy) for k in range(n)} for t, xy in spots.items()}
    return per, frames_, tracks, np.zeros(n, int)


# player 1 has the ball 1 s, a 0.5 s flicker to 9, back to 1 for 0.5 s, passes to 2 (1 s), 1-frame blip of 9
SEQ = [(1, "A", 30), (9, "B", 15), (1, "A", 15), (None, None, 10), (2, "A", 30), (9, "B", 1), (None, None, 20)]


def _touches(st, tid):
    return next(p["touches"] for p in st["players"] if p["id"] == tid)


def test_touch_episodes_same_as_passes_use():
    per, frames_, tracks, state = _setup(SEQ)
    eps = AN.touch_episodes(frames_, FPS)
    assert [(e[2], e[3]) for e in eps] == [(1, "A"), (2, "A")]   # flicker inside 1's spell is a sandwich -> merged; 1-frame blip < floor
    ps, _ = AN.passes(per, frames_, [], {}, {"A": True, "B": False}, FPS)
    assert len(ps) == 1 and ps[0]["from"] == 1 and ps[0]["to"] == 2


def test_touch_counts_and_min_length():
    eps = [[0, 29, 1, "A"], [40, 41, 9, "B"], [50, 70, 1, "A"]]
    assert AN.touch_counts(eps, FPS, 0) == {1: 2, 9: 1}
    assert AN.touch_counts(eps, FPS, 0.4) == {1: 2}          # 2 frames < 0.4 s


def test_stats_uses_contacts_and_old_switch(monkeypatch):
    per, frames_, tracks, state = _setup(SEQ)
    eps = AN.touch_episodes(frames_, FPS)
    old = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, {"A": True, "B": False})
    assert _touches(old, 1) == 2 and _touches(old, 9) == 2       # old: the flicker splits 1's spell and the blip adds 9 twice
    new = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, {"A": True, "B": False}, episodes_=eps)
    assert _touches(new, 1) == 1 and _touches(new, 2) == 1 and _touches(new, 9) == 0
    monkeypatch.setenv("IPANEMA_TOUCHES", "old")
    sw = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, {"A": True, "B": False}, episodes_=eps)
    assert [p["touches"] for p in sw["players"]] == [p["touches"] for p in old["players"]]


def test_player_without_contact_gets_zero():
    per, frames_, tracks, state = _setup([(1, "A", 30), (None, None, 30)])
    st = AN.stats(per, frames_, [], [], tracks, state, FPS, 105.0, 68.0, {"A": True, "B": False}, episodes_=AN.touch_episodes(frames_, FPS))
    assert _touches(st, 1) == 1 and _touches(st, 2) == 0 and _touches(st, 9) == 0


def test_env_min_length(monkeypatch):
    monkeypatch.setenv("IPANEMA_TOUCH_MIN_S", "0.4")
    try:
        importlib.reload(AN); assert AN.TOUCH_MIN_S == 0.4
        assert AN.touch_counts([[0, 1, 5, "A"], [10, 30, 5, "A"]], FPS) == {5: 1}
    finally:
        monkeypatch.delenv("IPANEMA_TOUCH_MIN_S"); importlib.reload(AN)
    assert AN.TOUCH_MIN_S == 0.0


def test_lab_truth_counts_spell_ends(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import n0_26c_touches as T
    p = tmp_path / "ev.csv"
    p.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
                 "Home,PASS,,1,1,0,3,0,Player3,Player5,0,0,0,0\nHome,BALL LOST,INTERCEPTION,1,9,0,9,0,Player5,,0,0,0,0\n"
                 "Away,RECOVERY,INTERCEPTION,1,9,0,9,0,Player20,,0,0,0,0\nAway,SHOT,ON TARGET,1,20,0,25,0,Player20,,0,0,0,0\n"
                 "Away,PASS,,2,1,0,3,0,Player20,Player21,0,0,0,0\n")
    assert T.truth_touches(str(p), 1) == {103: 1, 105: 1, 220: 1}
    fr = _frames([(103, "A", 5), (None, None, 20), (103, "A", 5), (220, "B", 3)])
    assert T.old_touches(fr, 25.0) == {103: 2, 220: 1}
