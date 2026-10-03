"""S3 (3 Oct): pasted balls - patch cutting, spot choice and pasting on synthetic grass."""
import numpy as np, cv2
from ipanema import pasteball as PB


def _grass(h=360, w=640, bgr=(70, 150, 90)):
    rng = np.random.default_rng(0)
    return np.clip(np.full((h, w, 3), bgr, float) + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)


def _ball_crop(d=10, S=32):
    c = _grass(S, S)
    cv2.circle(c, (S // 2, S // 2), d // 2, (235, 240, 240), -1)
    cv2.circle(c, (S // 2 + 1, S // 2 + 1), max(1, d // 5), (30, 30, 30), -1)          # a dark panel
    return c


def test_cut_patch_clean_ball():
    p = PB.cut_patch(_ball_crop(10))
    assert p is not None and 8 <= p["d"] <= 12
    assert p["alpha"].max() > 0.9 and p["alpha"].shape == p["rgb"].shape[:2]


def test_cut_patch_rejects_lines_and_shadows():
    c = _ball_crop(10); c[2:5, :] = 250                                                 # a line through the ring
    c[27:30, :] = 250; c[:, 2:5] = 250
    assert PB.cut_patch(c) is None
    s = _grass(32, 32); cv2.circle(s, (16, 16), 5, (20, 50, 30), -1)                    # dark blob, no bright panel
    assert PB.cut_patch(s) is None


def test_size_at_follows_player_heights():
    boxes = [[100, y - 0.2 * y, 120, y] for y in (100, 150, 200, 250, 300)]           # height = 0.2 * feet row
    assert abs(PB.size_at(boxes, 200, 360) - PB.BALL_PER_PLAYER * 40) < 0.5
    assert PB.size_at(boxes, 100, 360) < PB.size_at(boxes, 300, 360)
    assert 4 <= PB.size_at([], 0, 360) <= 10                                             # fallback


def test_spots_avoid_people_and_respect_size():
    fr = _grass(720, 1280)
    boxes = [[x, y - 0.12 * y, x + 0.05 * y, y] for x, y in ((200, 200), (600, 300), (900, 450), (300, 600), (1000, 650))]
    sp = PB.spots(fr, boxes, 8, np.random.default_rng(1))
    assert len(sp) >= 4
    b = np.array(boxes)
    for x, y, d, kind in sp:
        assert 4 <= d <= 8 and kind in ("open", "line", "feet")
        r = d / 2 + 1
        assert not np.any((x + r > b[:, 0]) & (x - r < b[:, 2]) & (y + r > b[:, 1]) & (y - r < b[:, 3]))


def test_paste_changes_only_the_spot_and_labels_it():
    fr = _grass(360, 640); p = PB.cut_patch(_ball_crop(10))
    out, box = PB.paste(fr, p, 300, 200, 6.0, np.random.default_rng(2))
    assert box is not None and box[0] < 300 < box[2] and box[1] < 200 < box[3]
    diff = np.abs(out.astype(int) - fr.astype(int)).sum(2)
    ys, xs = np.nonzero(diff > 25)
    assert len(xs) > 3 and np.all(np.abs(xs - 300) <= 20) and np.all(np.abs(ys - 200) <= 20)
    assert out[197:204, 297:304].mean(2).max() > fr[197:204, 297:304].mean(2).max() + 30   # white panel brighter than grass
    _, none = PB.paste(fr, p, 3, 3, 6.0, np.random.default_rng(2))                      # too close to the edge
    assert none is None


def test_paste_is_deterministic_by_seed():
    fr = _grass(); p = PB.cut_patch(_ball_crop(10))
    a, _ = PB.paste(fr, p, 300, 200, 6.0, np.random.default_rng(5))
    b, _ = PB.paste(fr, p, 300, 200, 6.0, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_paste_in_window_stays_inside_and_away_from_real_ball():
    fr = _grass(720, 1280); p = PB.cut_patch(_ball_crop(10))
    boxes = [[x, y - 0.12 * y, x + 0.05 * y, y] for x, y in ((200, 200), (600, 300), (900, 450), (300, 600), (1000, 650))]
    win = (0, 0, 640, 640); real = (320, 300); hits = 0
    for s in range(12):
        out, info = PB.paste_in_window(fr, boxes, real, win, [p], np.random.default_rng(s))
        if info is None: assert out is fr; continue
        hits += 1
        assert 24 <= info["x"] < 616 and 24 <= info["y"] < 616
        assert np.hypot(info["x"] - real[0], info["y"] - real[1]) >= 40
        diff = np.abs(out.astype(int) - fr.astype(int)).sum(2); ys, xs = np.nonzero(diff > 25)
        assert np.all(np.abs(xs - info["x"]) <= 20) and np.all(np.abs(ys - info["y"]) <= 20)
    assert hits >= 6
    out, info = PB.paste_in_window(fr, boxes, real, (1200, 700, 1280, 720), [p], np.random.default_rng(0))   # no room
    assert info is None and out is fr


def test_load_patches_skips_test_window():
    allp = PB.load_patches(".")
    some = PB.load_patches(".", skip_t=lambda t: t < 1e9)        # every SFK-BP crop left out: only the K1 grounds remain
    assert len(allp) > len(some) > 0 and all(q["sid"].startswith("k1:") for q in some)
