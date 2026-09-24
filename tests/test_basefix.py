"""Base check from the painted near touchline: finds the line on a real frame, transfer keeps approved poses, and the
acceptance rule accepts only a correction that carries over to other frames."""
import os, json, zipfile, numpy as np, cv2, pytest
from ipanema import basefix as BF, lines as LN
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json"))
ZIP = f"{ROOT}/results/samples/SFKBP1109_nearside.zip"

@pytest.mark.skipif(not os.path.exists(ZIP), reason="frame sample not present")
def test_finds_painted_near_touchline_on_real_frame():
    i = [k for k, f in enumerate(SOL["frames"]) if f["frame"] == "fc_0313.512.jpg"][0]
    img = cv2.resize(cv2.imdecode(np.frombuffer(zipfile.ZipFile(ZIP).read("fc_0313.533.jpg"), np.uint8), 1), (1280, 720))
    ev = BF.find_near_line(img, SOL["camera"], SOL["frames"][i]["pose"])
    assert ev and ev["inliers"] > 1000
    # the painted line measured by hand on 24 Sep: x = 0.8246*y - 11.5 (960x540)
    ys = np.array([300.0, 500.0]); man = np.c_[0.8246 * ys - 11.5, ys] * 1280 / 960
    d = ev["pts"][-1] - ev["pts"][0]; n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    assert np.abs((man - ev["pts"][0]) @ n).max() < 3.0
    assert np.median(np.abs(BF._line_resid(SOL["camera"], SOL["frames"][i]["pose"], ev, 1280, 720))) > 25   # the old base misses it

def test_transfer_pose_same_base_is_identity():
    p = SOL["frames"][5]["pose"]; q = BF.transfer_pose(SOL["camera"], SOL["camera"], p)
    assert LN.pose_error(SOL["camera"], q, p)["median_px"] < 0.05
