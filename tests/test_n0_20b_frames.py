"""20 Oct (N0-20b): entry picture sheets for the free runner, dry run on a stand-in clip (no download)."""
import json, os, subprocess, sys
import numpy as np, cv2

def test_sheets(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "c.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (480, 270))
    for k in range(100): w.write(np.full((270, 480, 3), k % 255, np.uint8))
    w.release()
    tiles = [{"t": 1.0 + 0.5 * j, "frame": 10 + 5 * j, "ball_px": [960, 540], "line_px": [[800, 400], [1400, 1100]]} for j in range(4)]
    tiles2 = [{"t": 9.5, "frame": 95, "ball_px": None, "line_px": None}, {"t": 10.5, "frame": 105, "ball_px": None, "line_px": None}]
    json.dump({"clip": "x", "fps": 10, "moments": [{"id": "m01", "team": "dark", "attacks": "right", "tiles": tiles},
                                                   {"id": "m02", "team": "light", "attacks": "left", "tiles": tiles2}]}, open(tmp_path / "m.json", "w"))
    env = dict(os.environ, LOCAL_CLIP=str(tmp_path / "c.avi"), N0_20B_OUT=str(tmp_path / "out"), N0_20B_MOMENTS=str(tmp_path / "m.json"))
    subprocess.run([sys.executable, "tools/n0_20b_frames.py"], check=True, env=env, capture_output=True)
    j = json.load(open(tmp_path / "out" / "sheets.json"))
    assert j["sheets"] == {"m01": 4, "m02": 1}                       # frame 105 is past the clip's end
    im = cv2.imread(str(tmp_path / "out" / "m01.jpg")); assert im.shape == (2 * 360, 3 * 640, 3)
    ring = im[168:174, 316:324].reshape(-1, 3).astype(int)            # top of the ball ring (1920 px -> 640 px: 960,540 -> 320,180)
    assert (ring[:, 2] - ring[:, 0]).max() > 60                       # red there, the plain grey frame has none
