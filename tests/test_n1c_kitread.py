import os, sys, json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n1c_kitread as K
import n1c_kitvote_run as R

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def patch_hsv(h_deg, s, v, n=20):
    hsv = np.zeros((n, n, 3), np.uint8); hsv[..., 0] = int(h_deg / 2); hsv[..., 1] = int(s * 255); hsv[..., 2] = int(v * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("hsv,cls", [
    ((0, 0.05, 0.95), "white"),        # sunlit white AIK shirt
    ((208, 0.45, 0.55), "white"),      # white shirt in the shade reads blue-grey
    ((0, 0.10, 0.12), "dark"),         # black SFK shirt
    ((355, 0.80, 0.70), "red"),        # red AIK keeper
    ((186, 0.60, 0.88), "blue"),       # light-blue referee
    ((110, 0.60, 0.55), "none"),       # grass only (box missed the player)
])
def test_classify_patch(hsv, cls):
    assert K.classify_patch(patch_hsv(*hsv), cv2, np)[0] == cls


def test_torso_box_above_the_foot_and_clipped():
    x0, y0, x1, y1 = K.torso_box([960, 770], (1080, 1920, 3))
    assert y1 < 770 and x0 < 960 < x1 and 20 < y1 - y0 < 50            # near player ~130 px tall -> torso ~30 px
    assert K.torso_box([5, 2], (1080, 1920, 3)) is None                   # foot at the top edge: no torso in the picture
    far, near = K.player_height_px(520), K.player_height_px(770)
    assert 55 <= far <= 65 and 120 <= near <= 140


def test_interp_row_moves_players_between_rows():
    fr = [{"t": 10.0, "players": [{"id": 1, "px": [100, 500]}, {"id": 2, "px": [0, 0]}], "possession": "A", "carrier": 1},
          {"t": 10.3, "players": [{"id": 1, "px": [130, 530]}], "possession": "B", "carrier": None}]
    r = K.interp_row(fr, [f["t"] for f in fr], 10.2)
    p1 = [p for p in r["players"] if p["id"] == 1][0]
    assert p1["px"] == pytest.approx([120, 520]) and r["possession"] == "B"
    assert [p for p in r["players"] if p["id"] == 2][0]["px"] == [0, 0]  # only in one row: kept as it was
    assert fr[0]["players"][0]["px"] == [100, 500]                       # the export rows are not changed


def test_wanted_rows_in_halves_every_step():
    fr = [{"t": t, "players": [{"id": 1}] if t != 3000.3 else []} for t in (100.0, 400.0, 400.3, 400.6, 400.9, 3000.3, 3300.0, 4000.0)]
    assert R.wanted_rows(fr, step=1) == [1, 2, 3, 4, 7]                # before kick-off, half-time and empty rows out
    assert R.wanted_rows(fr, step=2) == [1, 3, 7]


def test_run_reads_the_right_frames(tmp_path):
    """a stand-in video whose left half is white in frames 30-59 and black elsewhere; a player standing there must read
    white exactly at the rows inside that second"""
    v = str(tmp_path / "v.mp4"); fps = 30.0
    w = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for k in range(90):
        img = np.full((480, 640, 3), (40, 140, 40), np.uint8)
        img[200:480, 0:320] = 245 if 30 <= k < 60 else 15
        w.write(img)
    w.release()
    fr = [{"t": t, "players": [{"id": 7, "team": "A", "gk": False, "px": [160, 470]}]} for t in (0.5, 1.2, 1.5, 2.5)]
    reads, votes = R.run(v, fr, [0, 1, 2, 3], cv2, np, fps=fps, log=lambda *a: None)
    cls = [K.CLASSES[r[1][0][1]] for r in reads]
    assert [r[0] for r in reads] == [0.5, 1.2, 1.5, 2.5]
    assert cls == ["dark", "white", "white", "dark"]
    assert votes["7"][K.CLASSES.index("white")] == 2
