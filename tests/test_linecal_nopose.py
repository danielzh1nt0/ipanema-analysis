import json
from ipanema import linecal as LC

def test_rows_without_pose_are_skipped(tmp_path):
    src = json.load(open("calibration/SFKBP1109_lines_match.json")); rows = src["rows"][:60]
    rows[10] = dict(rows[10], pose=None, confident=False, why=["no lines"]); rows[11] = dict(rows[11], pose=None, confident=False)
    p = tmp_path / "x_lines_match.json"; json.dump({"camera": src["camera"], "rows": rows}, open(p, "w"))
    cal = LC.calibration_for_clip(str(p), 30 * 20, 30.0, 1920, 1080, offset_s=0.0, log=lambda *a: None)
    assert len(cal["H"]) == 600 and cal["L"] > 0
