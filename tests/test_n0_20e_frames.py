"""20 Oct (N0-20e): restart picture sheets for the free runner, dry run on a stand-in clip (no download), plus the moments
file the lab wrote (both clips, blind spot order, spots drawn per tile)."""
import json, os, subprocess, sys
import numpy as np, cv2


def test_sheets(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "c.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (480, 270))
    for k in range(100): w.write(np.full((270, 480, 3), 90, np.uint8))
    w.release()
    tiles = [{"t": 1.0 + 0.5 * j, "dt": -1.0 + 0.5 * j, "frame": 10 + 5 * j, "ball_px": [960, 540], "spot_px": [[300, 300], None]} for j in range(4)]
    tiles2 = [{"t": 9.5, "dt": 0.0, "frame": 95, "ball_px": None, "spot_px": [None, [1500, 900]]}, {"t": 10.5, "dt": 1.0, "frame": 105, "ball_px": None, "spot_px": [None, None]}]
    json.dump({"clips": {"x": "x/video.mp4", "y": "y/video.mp4"}, "moments": [{"id": "a01", "clip": "x", "tiles": tiles}, {"id": "b01", "clip": "y", "tiles": tiles2}]}, open(tmp_path / "m.json", "w"))
    env = dict(os.environ, LOCAL_CLIP=str(tmp_path / "c.avi"), N0_20E_OUT=str(tmp_path / "out"), N0_20E_MOMENTS=str(tmp_path / "m.json"))
    subprocess.run([sys.executable, "tools/n0_20e_frames.py"], check=True, env=env, capture_output=True)
    j = json.load(open(tmp_path / "out" / "sheets.json"))
    assert j["sheets"] == {"a01": 4, "b01": 1}                       # frame 105 is past the stand-in's end
    im = cv2.imread(str(tmp_path / "out" / "a01.jpg")).astype(int); assert im.shape == (2 * 360, 3 * 640, 3)
    c = im[100, 100]; assert c[0] > 180 and c[1] > 180 and c[2] < 120   # spot 1 (300,300 -> 100,100) is a cyan cross
    r = im[168:174, 316:324].reshape(-1, 3); assert (r[:, 2] - r[:, 0]).max() > 60   # red ball ring
    im2 = cv2.imread(str(tmp_path / "out" / "b01.jpg")).astype(int)
    m = im2[300, 500]; assert m[0] > 180 and m[2] > 180 and m[1] < 120  # spot 2 (1500,900 -> 500,300) is a magenta cross


def test_moments_file_blind_and_complete():
    M = json.load(open("results/qa/n0-20e/moments.json")); K = json.load(open("results/qa/n0-20e/moments_key.json"))
    assert set(M["clips"]) == {"SFKBP1109_s1200", "p15u-vs-aik-2026-09-21-bd09_s2520"}
    ids = [m["id"] for m in M["moments"]]; assert sorted(ids) == sorted(K)
    assert {K[i]["spot1"] for i in ids} == {"old", "new"}                 # both orders used
    for m in M["moments"]:
        assert "old" not in json.dumps(m) and "new" not in json.dumps(m)   # nothing on the sheet says which window
        assert all(len(t["spot_px"]) == 2 for t in m["tiles"]) and len(m["tiles"]) == 15


def test_zoom_crops_around_spots(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "c.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1920, 1080))
    for k in range(20): w.write(np.full((1080, 1920, 3), 90, np.uint8))
    w.release()
    tiles = [{"t": 1.0, "dt": 0.0, "frame": 10, "ball_px": None, "spot_px": [[1500, 500], [1700, 500]]}]
    json.dump({"clips": {"x": "x/video.mp4"}, "moments": [{"id": "z01", "clip": "x", "tiles": tiles}]}, open(tmp_path / "m.json", "w"))
    env = dict(os.environ, LOCAL_CLIP=str(tmp_path / "c.avi"), N0_20E_OUT=str(tmp_path / "out"), N0_20E_MOMENTS=str(tmp_path / "m.json"))
    subprocess.run([sys.executable, "tools/n0_20e_frames.py", "--zoom", "800"], check=True, env=env, capture_output=True)   # as on the trigger line
    im = cv2.imread(str(tmp_path / "out" / "z01.jpg")).astype(int)
    # crop x 1120..1920 (clamped at the right edge), y 275..725 -> 0.8 px per 1920-px unit
    c = im[180, 304]; assert c[0] > 180 and c[1] > 180 and c[2] < 120   # spot 1 at (1500,500) -> (304, 180)
    m = im[180, 464]; assert m[0] > 180 and m[2] > 180 and m[1] < 120   # spot 2 at (1700,500) -> (464, 180)
