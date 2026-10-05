import os, json, cv2, numpy as np
import importlib.util

def _load():
    spec = importlib.util.spec_from_file_location("v1_frames", os.path.join(os.path.dirname(__file__), "..", "tools", "v1_frames.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_sheets_from_local_video(tmp_path, monkeypatch):
    m = _load(); monkeypatch.setattr(m, "OUT", str(tmp_path / "out"))
    vid = str(tmp_path / "v.mp4"); w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920, 1080))
    for k in range(40):
        f = np.zeros((1080, 1920, 3), np.uint8); cv2.circle(f, (100 + 40 * k, 650), 8, (255, 255, 255), -1); w.write(f)
    w.release()
    spec = {"jobs": [{"name": "full", "src_key": vid, "times": [0.5, 1.0, 1.5, 2.0], "tile_w": 480, "per_sheet": 4, "cols": 2},
                     {"name": "crop", "src_key": vid, "times": [1.0, 1.2], "crop": [400, 400, 960, 540], "tile_w": 960, "per_sheet": 6, "cols": 2}]}
    idx = m.run(spec, base="")
    assert len(idx["full"]["sheets"]) == 1 and len(idx["crop"]["sheets"]) == 1
    s = cv2.imread(idx["full"]["sheets"][0]); assert s.shape[1] == 960 and s.shape[0] == 2 * 270
    c = cv2.imread(idx["crop"]["sheets"][0]); assert c.shape == (540, 1920, 3)
    # crop at t=1.0 (frame 10): ball at x=500 -> 100 px into the crop, y 650 -> 250; full resolution kept
    assert c[245:256, 93:108].min(axis=2).max() > 200 and c[245:256, 310:390].max() < 120
    assert json.load(open(os.path.join(m.OUT, "index.json")))["crop"]["crops"][0] == [400, 400, 960, 540]

def test_crop_clamped_inside_frame():
    m = _load(); f = np.full((1080, 1920, 3), 50, np.uint8)
    g = m.tile(f, 0.0, [1800, 1000, 960, 540], 960, "x"); assert g.shape == (540, 960, 3)

def test_moving_crop_follows_keyframes():
    m = _load(); job = {"centers": [[10.0, 960, 540], [12.0, 1160, 640]], "size": [960, 540]}
    assert m.crop_at(job, 9.0) == [480, 270, 960, 540]
    assert m.crop_at(job, 11.0) == [580, 320, 960, 540]
    assert m.crop_at(job, 13.0) == [680, 370, 960, 540]
    assert m.crop_at({"crop": [1, 2, 3, 4]}, 5.0) == [1, 2, 3, 4]
