"""N0-21e: the app's RF-DETR ball finder post-processing (ipanema/ballrf.py) - tiles, offsets, merging, the tile-edge
rule (IPANEMA_BALL_EDGE) and guesses scaled back to the video's own pixels. Stand-in detector, no rfdetr needed."""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, pytest
from ipanema import ballrf as BR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class D:
    def __init__(self, boxes, conf): self.xyxy = np.array(boxes, float).reshape(-1, 4); self.confidence = np.array(conf, float)


class Stub:
    """per tile index (0..7, row-major, as BR.tiles) -> list of (box in tile pixels, conf); same for every frame"""
    def __init__(self, per_tile): self.per_tile = per_tile; self.calls = 0
    def predict(self, crops, threshold):
        self.calls += 1; out = []
        for i, _ in enumerate(crops):
            t = self.per_tile.get(i % 8, []); out.append(D([b for b, _ in t], [c for _, c in t]))
        return out


def frame(): return np.zeros((1080, 1920, 3), np.uint8)


def test_tiles_cover_the_picture():
    t = BR.tiles(1920, 1080)
    assert t == [(0, 0), (427, 0), (853, 0), (1280, 0), (0, 440), (427, 440), (853, 440), (1280, 440)]
    cov = np.zeros((1080, 1920), bool)
    for x, y in t: cov[y:y + BR.CROP, x:x + BR.CROP] = True
    assert cov.all()


def test_cut_by_tile_inner_sides_only():
    # tile (0, 0): left and top are the picture's edge, right (640) and bottom (640) are inner sides
    assert not BR.cut_by_tile((0, 0, 10, 10), 0, 0, 1920, 1080)
    assert BR.cut_by_tile((630, 300, 639, 310), 0, 0, 1920, 1080)
    assert BR.cut_by_tile((300, 630, 310, 640), 0, 0, 1920, 1080)
    assert not BR.cut_by_tile((300, 300, 312, 312), 0, 0, 1920, 1080)
    # tile (1280, 440): right and bottom are the picture's edge, left and top inner
    assert not BR.cut_by_tile((630, 630, 640, 640), 1280, 440, 1920, 1080)
    assert BR.cut_by_tile((0, 300, 8, 310), 1280, 440, 1920, 1080)
    assert BR.cut_by_tile((300, 1, 310, 9), 1280, 440, 1920, 1080)


def test_offsets_dedup_order_and_batch(monkeypatch):
    monkeypatch.delenv("IPANEMA_BALL_EDGE", raising=False)
    # the same ball seen by tile 0 (0,0) and tile 1 (427,0) at picture (500, 200): kept once, the more confident
    m = Stub({0: [((494, 194, 506, 206), 0.4)], 1: [((67, 194, 79, 206), 0.7)], 6: [((100, 100, 110, 110), 0.2)]})
    out = BR.detect_many(m, [frame(), frame()])
    assert len(out) == 2 and out[0] == out[1] and m.calls == 1                 # one predict call for the whole batch
    g = out[0]
    assert len(g) == 2 and g[0][2] == pytest.approx(0.7) and g[0][:2] == pytest.approx((500, 200))
    assert g[1][:2] == pytest.approx((853 + 105, 440 + 105))                  # tile 6 offset
    assert [z[2] for z in g] == sorted([z[2] for z in g], reverse=True)


def test_at_most_20_guesses():
    m = Stub({4: [((10 + 30 * i, 100, 20 + 30 * i, 110), 0.1 + 0.01 * i) for i in range(25)]})
    g = BR.detect_many(m, [frame()], edge=False)[0]
    assert len(g) == 20 and g[0][2] == pytest.approx(0.34)


def test_edge_rule_off_by_default_keeps_cut_guess(monkeypatch):
    monkeypatch.delenv("IPANEMA_BALL_EDGE", raising=False)
    m = Stub({0: [((632, 300, 640, 312), 0.6)]})                              # cut by tile 0's right side, nothing else there
    assert BR.detect_many(m, [frame()])[0] == [pytest.approx((636, 306, 0.6))]


def test_edge_rule_on_drops_cut_junk_keeps_whole_ball(monkeypatch):
    monkeypatch.setenv("IPANEMA_BALL_EDGE", "1")
    # junk: cut by tile 0's right side (no ball there); ball: cut by tile 0's bottom, seen whole by tile 4 (0, 440)
    m = Stub({0: [((632, 300, 640, 312), 0.6), ((200, 632, 212, 640), 0.5)], 4: [((200, 190, 212, 202), 0.3)]})
    raw = []; g = BR.detect_many(m, [frame()], raw=raw)[0]
    assert len(g) == 1 and g[0] == pytest.approx((206, 636, 0.3))            # the whole view of the ball, at the same spot
    assert [z[3] for z in raw[0]] == [True, True, False]
    assert BR.detect_many(m, [frame()], edge=False)[0][0] == pytest.approx((636, 306, 0.6))   # explicit old
    monkeypatch.setenv("IPANEMA_BALL_EDGE", "0")
    assert len(BR.detect_many(m, [frame()])[0]) == 2                          # old: junk kept, ball once (cut copy, conf 0.5)


def test_merge_matches_detect_many():
    m = Stub({0: [((632, 300, 640, 312), 0.6), ((100, 100, 112, 112), 0.2)], 3: [((5, 5, 15, 15), 0.9)]})
    raw = []; o = BR.detect_many(m, [frame()], edge=False, raw=raw)[0]
    assert BR.merge(raw[0], False) == o and len(BR.merge(raw[0], True)) == len(o) - 1


def test_guesses_back_in_the_video_pixels(tmp_path):
    p = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    for _ in range(3): w.write(np.zeros((720, 1280, 3), np.uint8))
    w.release()
    m = Stub({0: [((94, 94, 106, 106), 0.5)]})                               # (100, 100) in the 1920x1080 frame
    out = BR.candidates_model(m, p, batch=2)
    assert sorted(out) == [0, 1, 2] and out[0][0] == pytest.approx((100 * 1280 / 1920, 100 * 720 / 1080, 0.5))
    assert BR._to_native([(10.0, 20.0, 0.1)], 1920, 1080) == [(10.0, 20.0, 0.1)]


def test_candidates_cached_reader_scales(tmp_path, monkeypatch):
    p = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 540))
    for _ in range(2): w.write(np.zeros((540, 960, 3), np.uint8))
    w.release()
    monkeypatch.setattr(BR, "load", lambda weights: Stub({7: [((10, 10, 20, 20), 0.4)]}))
    out = BR.candidates(p, "x", str(tmp_path / "c.pkl"), batch=4)
    assert out[1][0][:2] == pytest.approx(((1280 + 15) / 2, (440 + 15) / 2))
    assert os.path.exists(tmp_path / "c.pkl")


def test_tool_dry_run(tmp_path):
    clips = {}
    for name, n in (("s", 140), ("a", 530)):
        p = str(tmp_path / f"{name}.mp4"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (480, 270))
        for _ in range(n): w.write(np.full((270, 480, 3), 90, np.uint8))
        w.release(); clips[name] = p
    env = dict(os.environ, LOCAL_CLIP_SFK=clips["s"], LOCAL_CLIP_AIK=clips["a"], MAX_KEYS="3", FAKE_MODEL="1", PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "tools/n0_21e_balledge.py", str(tmp_path / "out")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.load(open(tmp_path / "out" / "edge_frames.json"))
    for c in d["totals"].values():
        assert c["frames"] == 3 and c["cut"] > 0 and c["dropped"] > 0 and c["new_n"] < c["old_n"]
    assert os.path.exists(tmp_path / "out" / "dropped_00.jpg")
