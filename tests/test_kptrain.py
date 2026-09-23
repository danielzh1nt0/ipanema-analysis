"""Training data from clicks: labels reproduce the clicked points, only in-view points are visible, splits are honest."""
import os, json, zipfile, tempfile, numpy as np, cv2
from ipanema.kptrain import build_dataset, project, KEYPOINT_NAMES
from ipanema.label import pitch_keypoints
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_from_solution():
    sol = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json")); d = tempfile.mkdtemp(); zips = {}
    for s in ("s1", "s2"):
        zp = f"{d}/{s}.zip"; z = zipfile.ZipFile(zp, "w")
        for fr in sol["frames"]:
            if fr["session"] == s: z.writestr(fr["frame"], cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))[1].tobytes())
        z.close(); zips[s] = zp
    y = build_dataset(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json", zips, f"{d}/ds")
    tr, va = os.listdir(f"{d}/ds/labels/train"), os.listdir(f"{d}/ds/labels/val")
    assert len(tr) + len(va) >= 50 and 5 <= len(va) <= 12 and not set(tr) & set(va)
    vals = np.array(open(f"{d}/ds/labels/train/{tr[0]}").read().split()[5:], float).reshape(-1, 3)
    assert vals.shape == (len(KEYPOINT_NAMES), 3) and set(vals[:, 2]) <= {0.0, 2.0} and ((vals[:, :2] >= 0) & (vals[:, :2] <= 1)).all()
    # the projected labels land within a few pixels of Daniel's own clicks for a clicked frame
    fr = sol["frames"][0]; clicks = json.load(open(f"{ROOT}/results/labels/SFKBP1109_points_{fr['session']}.json"))[fr["frame"]]["pairs"]
    kp = pitch_keypoints(); good = [n for n, u, v in clicks if "penalty spot" not in n]
    q = project(sol["camera"], fr["pose"], [kp[n] for n in good], 1280, 720); Q = np.array([[u, v] for n, u, v in clicks if "penalty spot" not in n], float)
    assert np.median(np.linalg.norm(q - Q, axis=1)) < 15
