"""N0-21a (21 Oct): player boxes cut by a tile's inner edge (FOLLOW_TILES far band ends at 65% of the picture).
In 37 saved RF-DETR tracking runs 15,399 feet sat exactly on that line (y 701 of 1080) and the 50 px below were nearly
empty; by eye the boxes end at the knees. drop_cut drops a cut box when another tile has the same person uncut."""
import os, sys, types, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import tracking as TR

W, H = 1920, 1080
RECTS = [(int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)) for x0, y0, x1, y1 in TR.FOLLOW_TILES]
BAND_BOTTOM = RECTS[1][3]                       # 702


def test_band_bottom_cut_box_dropped_when_whole_frame_has_the_player(monkeypatch):
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    full = [700, 610, 750, 760]                  # whole-frame tile: feet at 760
    cut = [701, 609, 751, BAND_BOTTOM - 0.5]     # left band tile: same person, box ends at the band edge
    keep = TR.drop_cut([full, cut], [0, 1], RECTS, W, H)
    assert keep.tolist() == [True, False]


def test_cut_box_alone_is_kept(monkeypatch):
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    keep = TR.drop_cut([[701, 609, 751, BAND_BOTTOM - 0.5]], [1], RECTS, W, H)
    assert keep.tolist() == [True]


def test_uncut_and_frame_edge_boxes_untouched(monkeypatch):
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    boxes = [[700, 400, 730, 480],               # inside the band, not touching its edges
             [701, 401, 731, 481],               # the same player from the whole frame (NMS merges these later)
             [0, 300, 30, 380],                  # left band tile, at the picture's left edge (a frame edge, not a cut)
             [1890, 900, 1920, 1080]]            # whole frame, bottom-right corner
    keep = TR.drop_cut(boxes, [1, 0, 1, 0], RECTS, W, H)
    assert keep.all()


def test_inner_side_edge_cut(monkeypatch):
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    x_edge = RECTS[1][2]                          # left band tile ends at 55% of the width
    full = [x_edge - 20, 400, x_edge + 15, 480]
    half = [x_edge - 20, 401, x_edge - 1, 479]   # left half of the player, cut by the tile's right edge
    keep = TR.drop_cut([full, half], [2, 1], RECTS, W, H)
    assert keep.tolist() == [True, False]


def test_partner_must_cover_the_cut_box(monkeypatch):
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    cut = [701, 609, 751, BAND_BOTTOM - 0.5]
    neighbour = [740, 600, 790, 760]             # another player overlapping a little
    keep = TR.drop_cut([cut, neighbour], [1, 0], RECTS, W, H)
    assert keep.tolist() == [True, True]


def test_switch_off_keeps_all(monkeypatch):
    monkeypatch.setenv("IPANEMA_TILE_CUT", "0")
    keep = TR.drop_cut([[700, 610, 750, 760], [701, 609, 751, BAND_BOTTOM - 0.5]], [0, 1], RECTS, W, H)
    assert keep.all()


def _fake_rfdetr(monkeypatch, per_tile, frame):
    """stands in for the rfdetr package: predict() returns per_tile[t] (boxes in TILE pixels, conf) for crop t"""
    sv = __import__("supervision")

    class _M:
        def predict(self, crops, threshold):
            out = []
            for c in crops:                                                   # which tile: the crop equals that region of the frame (RGB)
                t = next(i for i, (a, b, c_, d) in enumerate(RECTS) if c.shape[:2] == (d - b, c_ - a) and np.array_equal(c, frame[b:d, a:c_, ::-1]))
                bx = per_tile.get(t, [])
                if not bx: out.append(sv.Detections.empty()); continue
                xy = np.array([b[:4] for b in bx], float); cf = np.array([b[4] for b in bx], float)
                out.append(sv.Detections(xyxy=xy, confidence=cf, class_id=np.ones(len(bx), int)))
            return out
    mod = types.SimpleNamespace(RFDETRSmall=_M, RFDETRMedium=_M)
    monkeypatch.setitem(sys.modules, "rfdetr", mod)
    import ipanema.ballrf as BR
    monkeypatch.setattr(BR, "autocast", contextlib.nullcontext)


def test_detect_batch_keeps_the_whole_player(monkeypatch):
    """the app's detector path end to end: the zoomed band tile is MORE confident about the cut box, which NMS used to keep"""
    try: import supervision, cv2  # noqa
    except ImportError: return
    a, b = RECTS[1][0], RECTS[1][1]
    per_tile = {0: [(700, 610, 750, 760, 0.55)],                                 # whole frame, frame pixels
                1: [(701 - a, 609 - b, 751 - a, BAND_BOTTOM - b - 0.5, 0.90)]}   # band tile, tile pixels, cut at its bottom
    f = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    _fake_rfdetr(monkeypatch, per_tile, f)
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    det, _ = TR.RFDetrPerson("small").detect_batch([f], 0.3, TR.FOLLOW_TILES)[0]
    assert len(det) == 1 and abs(det.xyxy[0, 3] - 760) < 1e-6                  # feet where they are
    monkeypatch.setenv("IPANEMA_TILE_CUT", "0")
    det0, _ = TR.RFDetrPerson("small").detect_batch([f], 0.3, TR.FOLLOW_TILES)[0]
    assert len(det0) == 1 and abs(det0.xyxy[0, 3] - (BAND_BOTTOM - 0.5)) < 1e-6   # old: feet on the band edge (IoU 0.61 > 0.5, cut box more confident)


def test_detect_batch_two_frames_tile_offsets(monkeypatch):
    """boxes map back to frame pixels per tile and per frame of the batch"""
    try: import supervision, cv2  # noqa
    except ImportError: return
    a, b = RECTS[2][0], RECTS[2][1]
    f = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    _fake_rfdetr(monkeypatch, {2: [(10, 20, 30, 60, 0.8)]}, f)
    monkeypatch.delenv("IPANEMA_TILE_CUT", raising=False)
    out = TR.RFDetrPerson("small").detect_batch([f, f], 0.3, TR.FOLLOW_TILES)
    assert len(out) == 2
    for det, names in out:
        assert names == {0: "person"} and det.xyxy.tolist() == [[a + 10, b + 20, a + 30, b + 60]]


def test_tool_dry_run(monkeypatch, tmp_path):
    """dry run of tools/n0_21a_tilecut.py (the free-runner job) with a stand-in detector on a real saved frame"""
    try: import supervision, cv2  # noqa
    except ImportError: return
    import glob, json, importlib.util
    fr = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/qa/p2/p15u-vs-spanga-2026-09-25_1159/raw_000*.jpg")))[:1]
    if not fr: return
    a, b = RECTS[1][0], RECTS[1][1]
    _fake_rfdetr(monkeypatch, {0: [(700, 610, 750, 760, 0.55), (100, 100, 120, 150, 0.6)], 1: [(701 - a, 609 - b, 751 - a, BAND_BOTTOM - b - 0.5, 0.90)]}, cv2.imread(fr[0]))
    spec = importlib.util.spec_from_file_location("n0_21a", os.path.join(os.path.dirname(TR.__file__), "..", "tools", "n0_21a_tilecut.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tot, rows = mod.main(str(tmp_path), fr)
    assert tot["frames"] == 1 and tot["old_on_edge"] == 1 and tot["new_on_edge"] == 0 and tot["moved"] == 1
    assert rows[0]["new"][3] == 760 and os.path.exists(tmp_path / "sheet.jpg")
    assert json.load(open(tmp_path / "changes.json"))["totals"]["moved"] == 1
