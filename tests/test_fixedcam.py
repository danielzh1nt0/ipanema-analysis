import json, numpy as np, cv2
from ipanema import fixedcam as FC

def _spec(tmp_path, noise=0.0):
    # a synthetic camera: a known homography, 8 pitch points projected through it
    Ht = np.array([[12.0, -3.0, 300.0], [1.0, 7.0, 150.0], [0.0, 0.002, 1.0]])
    pts = [(0, 0), (105, 0), (0, 68), (105, 68), (52.5, 34), (16.5, 13.84), (88.5, 54.16), (52.5, 0)]
    rng = np.random.default_rng(0); rows = []
    for x, y in pts:
        u, v, s = Ht @ [x, y, 1.0]; rows.append({"pitch_m": [x, y], "px": [u / s + rng.normal(0, noise), v / s + rng.normal(0, noise)], "name": f"{x},{y}"})
    p = tmp_path / "clip.json"; json.dump({"pitch": {"length": 105, "width": 68}, "image_size": [1920, 1080], "points": rows}, open(p, "w")); return p, Ht

def test_fit_recovers_the_camera(tmp_path):
    p, Ht = _spec(tmp_path); cal = FC.calibration(str(p), 10, 1920, 1080, log=lambda *a: None)
    H = cal["H"][3]; q = cv2.perspectiveTransform(np.float32([[[30.0, 20.0]]]), H)[0, 0]; u, v, s = Ht @ [30.0, 20.0, 1.0]
    assert abs(q[0] - u / s) < 0.5 and abs(q[1] - v / s) < 0.5 and cal["L"] == 105 and len(cal["H"]) == 10

def test_scaled_video_and_noise(tmp_path):
    p, Ht = _spec(tmp_path, noise=2.0); cal = FC.calibration(str(p), 2, 1280, 720, log=lambda *a: None)
    q = cv2.perspectiveTransform(np.float32([[[52.5, 34.0]]]), cal["H"][0])[0, 0]; u, v, s = Ht @ [52.5, 34.0, 1.0]
    assert abs(q[0] - u / s * 1280 / 1920) < 6 and abs(q[1] - v / s * 720 / 1080) < 6

def test_find_strips_piece_suffix(tmp_path):
    p, _ = _spec(tmp_path); assert FC.find("clip_c003", root=str(tmp_path)) == str(p) and FC.find("other", root=str(tmp_path)) is None
