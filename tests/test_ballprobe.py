import numpy as np
from ipanema import ballprobe as BP


def test_flip_back_and_tiles():
    assert BP.flip_back([(10.0, 5.0, 0.3)], 100) == [(89.0, 5.0, 0.3)]
    t = BP.tiles_2x2((1080, 1920, 3))
    assert len(t) == 4 and t[0][:2] == (0, 0) and t[-1][2:] == (1920, 1080)


def test_detect_tiled_maps_back_to_full_frame():
    frame = np.zeros((1080, 1920, 3), np.uint8); frame[900, 1700] = 255
    def fake(img):                                             # 'finds' the one bright pixel in whatever it is shown
        ys, xs = np.nonzero(img[:, :, 0]); return [(float(x), float(y), 0.02) for x, y in zip(xs, ys)]
    d = BP.detect_tiled(fake, frame)
    assert len(d) == 1 and d[0][:2] == (1700.0, 900.0)


def test_grade_and_summary():
    g = BP.grade([(0, 0, 0.9), (100, 100, 0.01)], (105, 98))
    assert g["rank"] == 1 and g["score"] == 0.01
    rows = [{"id": 1, "group": None, "truth": (105, 98), "a": g, "b": BP.grade([], (105, 98))},
            {"id": 2, "group": None, "truth": (5, 5), "a": BP.grade([(900, 900, 0.5)], (5, 5)), "b": BP.grade([(6, 6, 0.3)], (5, 5))},
            {"id": 3, "group": None, "truth": (5, 5), "a": BP.grade([], (5, 5)), "b": BP.grade([], (5, 5))},
            {"id": 4, "group": None, "truth": None, "a": BP.grade([(1, 1, 0.2)], None)}]
    s = BP.summarise(rows, ["a", "b"])
    assert s["ball_moments"] == 3 and s["variants"]["a"]["seen_only_below_cut"] == 1 and s["variants"]["b"]["seen_at_normal_cut"] == 1
    assert s["seen_by_any_variant"] == 2 and s["never_seen"] == [3]


def test_local_peaks_keeps_two_close_weak_peaks_apart():
    hm = np.zeros((100, 100), np.float32); hm[50, 40] = 0.03; hm[50, 60] = 0.02; hm[45:56, 40:61] += 0.012
    p = BP.local_peaks(hm, thr=0.01, nms=5)
    assert [(x, y) for x, y, _ in p[:2]] == [(40.0, 50.0), (60.0, 50.0)]


def test_run_with_stand_in_finders():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    moments = [{"id": "a", "frame": 5, "truth": (100, 100)}, {"id": "b", "frame": 6, "truth": None}, {"id": "c", "frame": 7, "truth": (1, 1)}]
    read3 = lambda k: None if k == 7 else (frame, frame, frame)
    rows = BP.run(moments, read3, {"low": lambda f3: [(101, 99, 0.01), (500, 500, 0.4)], "bad": lambda f3: 1 / 0}, log=lambda *a: None)
    assert len(rows) == 2 and rows[0]["low"]["rank"] == 1 and "error" in rows[0]["bad"] and rows[1]["low"]["has_ball"] is False
