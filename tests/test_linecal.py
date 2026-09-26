import os, json, numpy as np, cv2
from ipanema import linecal as LC, lines as LN
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_homography_matches_projection():
    cam = {"C": [53.3, 67.6, -4.8], "base_tilt": [0.0, 0.0]}; pose = [-1.7, 0.1, 0.003, 1480.0]
    H = LC.homography(cam, pose, 1920, 1080)
    P = np.array([[10, 10], [53, 32], [90, 60], [30, 5]], float)
    q1 = LN.project(cam, pose, P, 1920, 1080); q2 = cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)).reshape(-1, 2)
    assert np.abs(q1 - q2).max() < 0.05

def test_clip_from_match_rows():
    base, off, path = LC.find_rows("/nonexistent", "SFKBP1109_s1200")
    assert base == "SFKBP1109" and off == 1200.0
    cal = LC.calibration_for_clip(path, 300, 29.97, 1920, 1080, offset_s=off, log=lambda *a: None)
    assert len(cal["H"]) == 300 and all(cal["H"][k] is not None for k in cal["H"]) and 0.5 <= cal["coverage"] <= 1.0
    assert LC.find_rows("/nonexistent", "some-other-match") is None
