"""17 Oct (Kaggle join 7): a full match is graded on its own by-eye ball key when reference/<id>/ball_gt.json exists."""
import json, os
from ipanema import fullmatch as FM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_key_found_for_any_match(tmp_path):
    d = tmp_path / "reference" / "some-match"; d.mkdir(parents=True)
    (d / "ball_gt.json").write_text("{}")
    assert FM.ball_key_path(str(tmp_path), "some-match") == f"{tmp_path}/reference/some-match/ball_gt.json"


def test_no_key_gives_none(tmp_path):
    (tmp_path / "reference" / "other").mkdir(parents=True)
    assert FM.ball_key_path(str(tmp_path), "other") is None
    assert FM.ball_key_path(str(tmp_path), "missing") is None


def test_real_vallentuna_key_is_used_and_valid():
    p = FM.ball_key_path(ROOT, "p15u-vs-vallentuna-2026-10-03-6cce")
    assert p is not None
    gt = json.load(open(p))
    assert len(gt) >= 50 and all(k.isdigit() for k in gt)   # join 7 graded 55 frames


def test_kaggle_script_uses_helper():
    src = open(f"{ROOT}/kaggle/full_match.py").read()
    assert "gt_path = FM.ball_key_path(ROOT, M)" in src


def test_clip_key_remap_still_shifts_frames(tmp_path):
    src = tmp_path / "gt.json"; src.write_text(json.dumps({"10": [1, 2], "0": None}))
    out = json.load(open(FM.remap_gt(str(src), 100, 25.0, str(tmp_path / "o.json"))))
    assert out == {"2510": [1, 2], "2500": None}
