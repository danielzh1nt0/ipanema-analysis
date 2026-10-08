"""C3d: the far-paint cost of a camera row is low when the drawn far lines sit on the paint and high when the pose is off;
the free-runner job reads a video front to back and writes one line per posed second plus blind look pictures."""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import c3d_farcost as F
from ipanema import lines as LN, linecal as LC

def _camera_pose():
    d = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "SFKBP1109_lines_match.json")))
    r = next(r for r in d["rows"] if r.get("pose") is not None and r["t"] >= 1500 and LC.brave(r))
    return d["camera"], np.array(r["pose"], float)

def _frame(camera, pose, w=1280, h=720):
    img = np.zeros((h, w, 3), np.uint8); img[:] = (40, 140, 40)
    for seg in LN.projected_segments(camera, pose, w, h).values():
        for x0, y0, x1, y1 in seg: cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (235, 235, 235), 3)
    return img

def test_far_cost_good_vs_off():
    cam, pose = _camera_pose(); far = F.model_pts(cam, 106.0, 64.0)[0]; img = _frame(cam, pose)
    good, n = F.far_cost(img, cam, pose, far); off, _ = F.far_cost(img, cam, pose + [0.03, 0.02, 0, 0], far)
    assert n >= 30 and good is not None and good < 1.5 and off > 6

def test_run_dry(tmp_path):
    cam, pose = _camera_pose(); fps = 5.0; d = tmp_path / "m"; os.makedirs(d)
    vw = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for k in range(int(10 * fps)): vw.write(_frame(cam, pose))
    vw.release()
    rows = [{"t": float(t), "pose": list(pose + ([0.03, 0.02, 0, 0] if t % 2 else 0)), "trusted": True, "inplay": True, "row_cost": 1.0, "why": []} for t in range(10)]
    json.dump({"match": "m", "src_key": "m/video.mp4", "camera": cam, "rows": rows}, open(d / "sample.json", "w"))
    os.environ["VIDEO"] = str(tmp_path / "v.mp4")
    try: assert F.run(str(d), n_per_bin=3) == 10
    finally: del os.environ["VIDEO"]
    res = json.load(open(d / "costs.json"))["rows"]; key = json.load(open(d / "look_key.json"))
    good = [r["far"] for r in res if r["t"] % 2 == 0]; bad = [r["far"] for r in res if r["t"] % 2]
    assert max(good) < 2.0 < 6.0 < min(bad)
    assert all(os.path.exists(d / "look" / f"{i}.jpg") for i in key) and len(key) >= 4

def test_veto_separation_and_shade_frame():
    import c3d_veto as V
    rows = [{"grade": "o", "s": 6.0}, {"grade": "o", "s": 8.0}, {"grade": "g", "s": 1.0}, {"grade": "g", "s": 7.0}, {"grade": "r", "s": 6.5}, {"grade": "?", "s": 9.0}]
    s = V.separation(rows, "s")
    assert s["thr"] == 6.0 and s["good_vetoed"] == 1 and s["good"] == 2 and s["rough_vetoed"] == 1
    assert V.separation([{"grade": "g", "s": 1.0}], "s") is None
    cam, pose = _camera_pose(); far, near = V.model_pts(cam, 106.0, 64.0); img = _frame(cam, pose)
    sc = V.scores(img, cam, pose, far, near)
    assert sc["far_shade"] < 1.5 and sc["move_px"] < 3
