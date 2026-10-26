"""N0-26b (26 Oct): kick-off line-ups as a direction rule (P.kickoff_spells / P.direction_from_kickoffs). At a kick-off every
player stands in his own half, so the half a team lines up in is the half it defends. Rows are [id, team, metres, feet px,
box, keeper]."""
import numpy as np
from ipanema import possession as P

L, W, FPS = 105.0, 68.0, 10.0
q = lambda *a: None


def lineup(a_left, n=6, jitter=0.0, k=0):
    rng = np.random.default_rng(k)
    xa = [L / 2 - 5 - 4 * i for i in range(n)]; xb = [L / 2 + 5 + 4 * i for i in range(n)]
    if not a_left: xa, xb = xb, xa
    rows = [[100 + i, "A", np.array([x + rng.normal(0, jitter), 30.0]), None, None, False] for i, x in enumerate(xa)]
    rows += [[200 + i, "B", np.array([x + rng.normal(0, jitter), 30.0]), None, None, False] for i, x in enumerate(xb)]
    return rows


def mixed(k):
    """open play: both teams spread over both halves"""
    return [[100 + i, "A", np.array([20.0 + 13 * i, 30.0]), None, None, False] for i in range(6)] + \
           [[200 + i, "B", np.array([26.0 + 13 * i, 30.0]), None, None, False] for i in range(6)]


def test_line_up_held_gives_the_defended_half():
    per = {k: lineup(True, k=k) for k in range(20)}
    per.update({k: mixed(k) for k in range(20, 200)})
    ballm = {k: np.array([L / 2, W / 2]) for k in range(20)}
    sp = P.kickoff_spells(per, ballm, FPS, L, W)
    assert len(sp) == 1 and sp[0]["a_left"] is True
    assert P.direction_from_kickoffs(per, ballm, FPS, L, W, log=q) == {"A": True, "B": False}   # A defends left -> attacks right


def test_other_way_round_and_unseen_ball_is_fine():
    per = {k: lineup(False, jitter=0.8, k=k) for k in range(20)}
    assert P.direction_from_kickoffs(per, {}, FPS, L, W, log=q) == {"A": False, "B": True}


def test_ball_far_from_the_spot_is_not_a_kickoff():
    per = {k: lineup(True, k=k) for k in range(20)}
    ballm = {k: np.array([L / 2 + 30, W / 2]) for k in range(20)}
    assert P.kickoff_spells(per, ballm, FPS, L, W) == []
    assert P.direction_from_kickoffs(per, ballm, FPS, L, W, log=q) is None


def test_too_short_too_few_or_open_play_gives_nothing():
    short = {k: lineup(True, k=k) for k in range(5)}                          # 0.4 s < 1 s
    assert P.kickoff_spells(short, {}, FPS, L, W) == []
    few = {k: [r for r in lineup(True, k=k) if r[0] % 100 < 3] for k in range(20)}   # 3 per team < 4
    assert P.kickoff_spells(few, {}, FPS, L, W) == []
    assert P.kickoff_spells({k: mixed(k) for k in range(100)}, {}, FPS, L, W) == []


def test_players_on_halfway_and_keepers_do_not_spoil_it():
    per = {}
    for k in range(20):
        rows = lineup(True, k=k)
        rows.append([300, "B", np.array([L / 2 - 0.5, 30.0]), None, None, False])    # kicker's opponent on the line: ignored
        rows.append([301, "A", np.array([L - 2.0, 30.0]), None, None, True])        # a keeper on the wrong side: keepers ignored
        per[k] = rows
    assert P.direction_from_kickoffs(per, {}, FPS, L, W, log=q) == {"A": True, "B": False}


def test_spells_vote_and_a_tie_is_no_answer():
    per = {}
    for k in range(20): per[k] = lineup(True, k=k)
    for k in range(20, 400): per[k] = mixed(k)
    for k in range(400, 420): per[k] = lineup(False, k=k)
    assert len(P.kickoff_spells(per, {}, FPS, L, W)) == 2
    assert P.direction_from_kickoffs(per, {}, FPS, L, W, log=q) is None
    for k in range(800, 820): per[k] = lineup(True, k=k)
    for k in range(420, 800): per.setdefault(k, mixed(k))
    assert P.direction_from_kickoffs(per, {}, FPS, L, W, log=q) == {"A": True, "B": False}


def test_one_short_gap_does_not_split_a_spell():
    per = {k: lineup(True, k=k) for k in range(30)}
    per[12] = mixed(12); per[13] = mixed(13)                                   # 0.2 s of mixed reading inside the line-up
    sp = P.kickoff_spells(per, {}, FPS, L, W)
    assert len(sp) == 1 and sp[0]["k0"] == 0 and sp[0]["k1"] == 29


def _export_frames(a_left, n=30, stride=0.3):
    fr = []
    for k in range(n):
        ps = [{"id": r[0], "team": r[1], "m": [float(r[2][0]), float(r[2][1])]} for r in lineup(a_left, k=k)]
        fr.append({"t": 100 + k * stride, "players": ps, "ball": {"m": [L / 2, W / 2]}})
    return fr


def test_upload_check_flags_a_direction_the_line_ups_contradict():
    from ipanema import uploadcheck as U
    md = {"attack_right": {"A": True, "B": False}}
    ok = U.check_direction(md, _export_frames(True), L, W)
    bad = U.check_direction(md, _export_frames(False), L, W)
    none = U.check_direction(md, [{"t": 1.0 + i, "players": [], "ball": None} for i in range(5)], L, W)
    assert ok["ok"] is True and bad["ok"] is False and none["ok"] is None
    assert U.PART["direction"] == "direction"
