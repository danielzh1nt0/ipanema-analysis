"""Full-match logic: a simulated camera path with a blackout, a deep zoom and a fast pan must come out mostly confident,
the blackout unsure, and no confident-but-wrong checkpoint."""
import json, os, numpy as np
from ipanema import lines as LN, matchcal as MC
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = json.load(open(f"{ROOT}/calibration/panorama/SFKBP1109_clicks_solution.json")); CAM = SOL["camera"]

def test_simulated_stretch():
    def true_pose(t): return [np.radians(-150 + t * 1.5 + (15 if t >= 22 else 0)), np.radians(8), 0.0, 2600.0 if 10 <= t < 14 else 1000.0]
    predict = lambda img: (np.zeros((360, 640), np.uint8) if 26 <= predict.t < 29 else np.where(LN.render_mask(CAM, true_pose(predict.t), 640, 360) == 255, 0, LN.render_mask(CAM, true_pose(predict.t), 640, 360)).astype(np.uint8))
    class F:
        def __iter__(self):
            for t in MC.frame_times(32, 1.0): predict.t = t; yield t, np.zeros((360, 640, 3), np.uint8)
    rows, _ = MC.run_chunk(F(), CAM, predict, fps=1.0, anchor_s=5.0, snap=False)
    conf = [r for r in rows if r["confident"]]; assert len(conf) >= 0.7 * len(rows), MC.summarize(rows, [])
    assert all(not r["confident"] for r in rows if 26 <= r["t"] < 29)                       # blackout: never guessed
    assert all(LN.pose_error(CAM, r["pose"], true_pose(r["t"]))["median_px"] < 10 for r in conf)   # confident means right
