"""N0 19 Oct: the export's kit label must match the pipeline convention (A = darker kit)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipanema import export as EX
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import team_letter_check as TLC


def test_convention_constant():
    assert EX.TEAM_KITS == {"dark": "A", "light": "B"}


def test_export_writes_dark_is_A(tmp_path):
    n, fps, L, W = 20, 10.0, 105.0, 68.0
    per = [[(i, "A" if i < 7 else "B", (50.0, 30.0), (100.0, 100.0), None, False) for i in range(14)] for _ in range(n)]
    fr = [{"carrier": None, "pressure_m": None, "near_opps": None} for _ in range(n)]
    H = {k: np.eye(3) for k in range(n)}
    EX.write(str(tmp_path), "m1", str(tmp_path / "none.mp4"), {"fps": fps, "width": 1920, "height": 1080}, per, fr, {}, {}, [2] * n, H, L, W,
             {"A": True, "B": False}, 0.5, [], [], [], [], {}, [{"A": None, "B": None}] * n, {"metrics": {"shots": [], "high_turnovers": []}, "teams": []}, None,
             {"ball_grade": None, "ball_reliable": False}, log=lambda *a: None, copy_video=False, make_zip=False)
    md = json.load(open(tmp_path / "matches" / "m1" / "match_data.json"))
    assert md["teams"] == {"dark": "A", "light": "B"}


def test_score_counts_both_namings():
    frames = [{"possession": p} for p in ("A", "A", "B", None, "B")]
    key = [{"t": 0, "truth": "dark"}, {"t": 1, "truth": "dark"}, {"t": 2, "truth": "white"}, {"t": 3, "truth": "white"},
           {"t": 4, "truth": "dark"}, {"t": 4, "truth": "loose"}, {"t": 99, "truth": "dark"}]
    s = TLC.score(frames, 1.0, key)
    assert s["A=dark"] == (3, 1, 1) and s["A=light"] == (1, 3, 1)


def test_repo_sfkbp_clip_says_A_is_dark():
    """the app's SFK-BP clip (repo copy) graded on the by-eye key in kit colours: A = dark wins clearly"""
    if not (os.path.exists(TLC.MD) and os.path.exists(TLC.KEY)):
        import pytest; pytest.skip("repo copy not here")
    s = TLC.main()
    assert s["A=dark"][0] >= 30 and s["A=dark"][1] <= 8
    assert s["A=dark"][0] - s["A=dark"][1] > s["A=light"][0] - s["A=light"][1] + 20
