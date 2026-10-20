"""20 Oct (N0-20a part 2): the picture-sheet script for the AIK clip, dry run on a stand-in clip (no download):
default = 12 frames every 2.5 s, 3 per row; 'dense' = 1 s apart, 960 px, 2 per row, stops at the clip's end."""
import json, os, subprocess, sys
import numpy as np, cv2

def _clip(path, secs=300, fps=2.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 180))
    for k in range(int(secs * fps)): w.write(np.full((180, 320, 3), k % 255, np.uint8))
    w.release()

def _run(tmp, *args):
    env = dict(os.environ, LOCAL_CLIP=str(tmp / "c.avi"), N0_20A_OUT=str(tmp / "out"))
    for k in ("N0_20A_EVENTS", "N0_20A_STEP", "N0_20A_W", "N0_20A_COLS", "N0_20A_N"): env.pop(k, None)
    subprocess.run([sys.executable, "tools/n0_20a_frames.py", *args], check=True, env=env, capture_output=True)
    return json.load(open(tmp / "out" / "sheets.json"))

def test_default_sheets(tmp_path):
    _clip(tmp_path / "c.avi")
    j = _run(tmp_path)
    assert j["sheets"] == {"shot_012": 12, "shot_095": 12, "goal_242": 12, "after_goal_265": 12} and j["step"] == 2.5
    assert cv2.imread(str(tmp_path / "out" / "goal_242.jpg")).shape == (4 * 360, 3 * 640, 3)

def test_dense_preset(tmp_path):
    _clip(tmp_path / "c.avi", secs=300.5)
    j = _run(tmp_path, "dense")
    assert j["step"] == 1.0 and j["tile_w"] == 960
    assert j["sheets"]["dense_254"] == 12 and j["sheets"]["dense_290"] == 11      # 290..300 s, the clip ends at 300.5 s; black padding not counted
    assert cv2.imread(str(tmp_path / "out" / "dense_254.jpg")).shape == (6 * 540, 2 * 960, 3)
