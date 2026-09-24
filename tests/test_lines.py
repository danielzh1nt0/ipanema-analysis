"""Line-class calibration: labels sit where Daniel clicked, the pose fit recovers verified poses from line masks with no
starting guess, and it refuses to be confident when the lines can't place the frame."""
import os, json, numpy as np
from ipanema import lines as LN
from ipanema.label import pitch_keypoints
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json")); CAM = SOL["camera"]

def test_labels_pass_through_clicked_corners():
    """box / goal-area corners Daniel clicked lie on the rendered lines of their classes (1280 frame)"""
    kp = pitch_keypoints(); near = []
    for fr in SOL["frames"][:20]:
        m = LN.render_mask(CAM, fr["pose"], 1280, 720)
        clicks = json.load(open(f"{ROOT}/results/labels/SFKBP1109_points_{fr['session']}.json"))[fr["frame"]]["pairs"]
        for n, u, v in clicks:
            if "area" not in n: continue
            y0, x0 = int(v), int(u); win = m[max(0, y0 - 12):y0 + 13, max(0, x0 - 12):x0 + 13]
            near.append(((win > 0) & (win != LN.IGNORE)).any())
    assert len(near) > 20 and np.mean(near) > 0.9, (len(near), np.mean(near))

def test_mirror_symmetric_classes():
    """a mirrored view gets the same class names: every class appears at both ends"""
    S = LN.class_segments()
    for k in (3, 6, 7, 8, 9, 10):
        xs = np.array([[a[0], b[0]] for a, b in S[k]]).ravel(); assert xs.min() < 20 and xs.max() > 86, LN.CLASSES[k]

def test_fit_recovers_clicked_poses_cold():
    for fr in SOL["frames"][::19]:
        p, info = LN.fit_pose(LN.render_mask(CAM, fr["pose"], 640, 360), CAM)
        assert LN.pose_error(CAM, p, fr["pose"])["median_px"] < 3.0 and info["confident"], (fr["frame"], info)

def test_not_confident_from_one_line_family():
    """only the far touchline visible: many views explain it -> must not claim confidence"""
    fr = SOL["frames"][2]; m = LN.render_mask(CAM, fr["pose"], 640, 360); m[(m != 2)] = 0
    p, info = LN.fit_pose(m, CAM); assert not info["confident"], info
