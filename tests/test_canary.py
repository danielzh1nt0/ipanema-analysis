from ipanema.fullmatch import canary_index, canary_ok, plan

def test_canary_index_prefers_first_half():
    p = plan(184725, 29.97)
    assert len(p) == 21 and canary_index(p, list(range(21))) == 4
    assert canary_index(p, [0, 1, 2, 16, 17]) == 2 and canary_index(p, []) is None

def test_canary_decisions():
    good = ["calibration: from panorama, 8992/8992 frames registered", "SFKBP1109_c004: 8992 frames, 12.6 players/frame"]
    assert canary_ok(good, True)[0]
    assert not canary_ok(["mosaic calibration failed: ValueError('matmul')", "SFKBP1109_c004: 8992 frames, 12.6 players/frame"], True)[0]
    assert not canary_ok(["calibration: 8992 frames, keypoints on 39%", "x: 8992 frames, 12.0 players/frame"], True)[0]
    assert canary_ok(["calibration: 8992 frames, keypoints on 39%", "x: 8992 frames, 12.0 players/frame"], False)[0]
    assert not canary_ok(["calibration: from panorama, 8992/8992 frames registered", "x: 8992 frames, 3.6 players/frame"], True)[0]
    assert not canary_ok(["PIECE FAILED: Traceback ..."], True)[0]
