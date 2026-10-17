"""F1 relabel stage (17 Oct, main session): kaggle/full_match.py STAGE=relabel re-reads every track's kit and writes
overrides/<match>_teams.json keyed by the JOINED ids; the join applies it with FM.apply_team_override(..., from_k=True)."""
import numpy as np
from ipanema import fullmatch as FM


def _row(i, team):
    return [i, team, 100.0, 200.0, [0.0, 0.0, 10.0, 20.0]]


def test_from_k_moves_neither_team_tracks():
    per = {0: [_row(1, "K"), _row(2, "A")], 1: [_row(1, "K"), _row(2, "A")]}
    old = {g: [list(r) for r in rows] for g, rows in per.items()}
    # default (K3c behaviour): a K row is never touched, an A row is
    assert FM.apply_team_override(per, {"1": "A", "2": "B"}) == 2
    assert [r[1] for r in per[0]] == ["K", "B"]
    # relabel behaviour: the K track joins the team its reads say
    per = old
    assert FM.apply_team_override(per, {"1": "A", "2": "B"}, from_k=True) == 4
    assert [r[1] for r in per[0]] == ["A", "B"] and [r[1] for r in per[1]] == ["A", "B"]


def test_from_k_leaves_unlisted_and_same_team_rows():
    per = {0: [_row(5, "K"), _row(6, "B"), _row(7, "A")]}
    assert FM.apply_team_override(per, {"6": "B"}, from_k=True) == 0        # same team: no change counted
    assert [r[1] for r in per[0]] == ["K", "B", "A"]                         # unlisted tracks keep their label


def test_relabel_ids_match_the_join():
    """the relabel stage computes the joined id as r[0] + i * ID_STRIDE; FM.join must give the same ids, so the
    override hits the same track in both places even when two pieces reuse a tracker id"""
    fps = 30.0; n = 600; m = 300
    plan_ = FM.plan(n, fps, piece_s=10)                                      # two pieces of 10 s = 300 frames
    assert [p["i"] for p in plan_] == [0, 1]
    H = np.eye(3, dtype=np.float32)
    def piece(team):
        return {"per": {k: [_row(3, team)] for k in range(m)}, "H": {k: H for k in range(m)}, "cands": {}, "coverage": 1.0,
                "frozen": 0, "L": 105, "W": 68, "width": 1920, "height": 1080, "dark_share": None, "strips": None}
    pieces = [piece("K"), piece("A")]
    relabel_ids = {}
    for p, pl in zip(pieces, plan_):                                         # same mapping as kaggle/full_match.py relabel
        for k, rows in p["per"].items():
            for r in rows:
                relabel_ids[int(r[0]) + pl["i"] * FM.ID_STRIDE if r[0] >= 0 else int(r[0])] = r[1]
    per, *_ = FM.join(pieces, plan_, n, fps)
    joined = {r[0]: r[1] for rows in per.values() for r in rows}
    assert joined == relabel_ids == {3: "K", FM.ID_STRIDE + 3: "A"}
    # the override written from those ids flips only piece 0's K track
    assert FM.apply_team_override(per, {"3": "B"}, fps, from_k=True) == m
    assert {r[1] for g in range(m) for r in per[g]} == {"B"} and {r[1] for g in range(m, n) for r in per[g]} == {"A"}
