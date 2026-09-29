import json, numpy as np, pytest
from ipanema import k1crops as K


def test_checked_balls_only_yes_match_ball():
    sp = K.checked_balls("results/review", "p15u-vs-spanga-2026-09-25")
    a = json.load(open("results/review/p15u-vs-spanga-2026-09-25_clicks/answers.json"))
    assert len(sp) == 12 and all(a["answers"][c["id"]] == "yes" for c in sp)
    assert not {c["id"] for c in sp} & set(a["spare_ball"])
    it = json.load(open("results/review/p15u-vs-spanga-2026-09-25_clicks/items.json"))
    assert sp[0]["frame"] == it[int(sp[0]["id"][-2:])]["frame"]                       # id = prefix + index into items
    assert sum(len(K.checked_balls("results/review", m)) for m in K.GOOD) == 19 + 20 + 12 + 14
    with pytest.raises(ValueError): K.checked_balls("results/review", "p15u-vs-reymersholm-2026-09-18")


def test_follow_stops_when_unsure_or_lost():
    peaks = {0: [(100, 100, 0.5), (300, 300, 0.9)], 1: [(105, 100, 0.4)], 2: [(110, 100, 0.4), (112, 104, 0.4)],   # dk 2: two about as near
             -1: [(95, 100, 0.3)], -2: [], -3: [], -4: [(80, 100, 0.9)]}                                          # lost after 2 misses
    f = K.follow(peaks, (101, 99), 4)
    assert f[0] == (100, 100, 0.5) and f[1] == (105, 100, 0.4) and 2 not in f
    assert -1 in f and -4 not in f
    g = K.follow({0: [(300, 300, 0.9)]}, (100, 100), 3)
    assert g == {0: (100.0, 100.0, -1.0)}                                                                        # no peak on the spot: the spot


def test_crops_for_check_labels():
    fr = {dk: np.full((200, 400, 3), 80, np.uint8) for dk in range(-3, 4)}
    pk = {dk: [(100 + 4 * dk, 100, 0.6), (300, 150, 0.8), (120 + 4 * dk, 104, 0.2)] for dk in range(-2, 3)}
    X, meta, fol = K.crops_for_check(fr, pk, {"frame": 1000, "xy": (100, 100)}, "m", 2)
    assert set(m[8] for m in meta) == {-2, 0, 2}                                                                  # every 2nd frame
    assert [m[6] for m in meta if m[8] == 0] == [1, 0]                                                            # the 10-30 px peak is left out
    assert meta[0][0] == "m:998" and len(X) == len(meta) and X[0].shape == (3, 32, 32, 3)


def test_follow_template_follows_moving_ball_and_stops_when_it_vanishes():
    import cv2
    rng = np.random.default_rng(0); fr = {}
    for dk in range(-6, 7):
        f = rng.integers(60, 90, (300, 400, 3)).astype(np.uint8)
        if dk <= 3: cv2.circle(f, (200 + 6 * dk, 150), 5, (240, 240, 240), -1); cv2.circle(f, (200 + 6 * dk, 153), 2, (40, 40, 40), -1)
        fr[dk] = f
    out = K.follow_template(fr, (200, 150), 6)
    assert all(abs(out[dk][0] - (200 + 6 * dk)) <= 1 and abs(out[dk][1] - 150) <= 2.5 for dk in range(-6, 4))   # centre pulled 2 px off the dark patch
    assert 4 not in out                                                                                         # ball gone: stop
    X, meta, fol = K.crops_for_check(fr, {0: [(201, 151, 0.7), (350, 250, 0.4)]}, {"frame": 50, "xy": (200, 150)}, "m", 6, how="template")
    assert fol[0] == (201.0, 151.0, 0.7) and fol[2][2] == -1.0 and sum(m[6] for m in meta) == len([d for d in fol if d % 2 == 0 and abs(d) <= 5])


def test_follow_template_moves_start_onto_ball_and_does_not_ride_a_line():
    import cv2
    rng = np.random.default_rng(1); fr = {}
    for dk in range(-4, 5):
        f = rng.integers(60, 90, (300, 400, 3)).astype(np.uint8); cv2.line(f, (0, 160), (399, 160), (235, 235, 235), 3)
        if dk <= 0: cv2.circle(f, (200 + 15 * dk, 166), 5, (240, 240, 240), -1)       # ball on the line, then kicked out of view
        fr[dk] = f
    out = K.follow_template(fr, (200, 160), 4)                                       # check 6 px above the ball
    assert abs(out[0][0] - 200) <= 1 and abs(out[0][1] - 166) <= 1
    assert -2 in out and 1 not in out                                                # no ball after dk 0: stop, don't slide along the line
    assert K.blobness(fr[0], (200, 166)) > 2 * K.blobness(fr[0], (300, 160))
