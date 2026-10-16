"""F1 (16 Oct): tools/export_qa.py grades a Kaggle export against the by-eye keys (ball px, who has the ball) and
finds the export/<match>/ layout the Kaggle workflow passes it."""
import json, os, subprocess, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import export_qa as Q  # noqa: E402


def _export(folder, frames_main, frames_chunk, summary=True):
    os.makedirs(folder, exist_ok=True)
    md = {"fps": 10.0, "frames": frames_main, "frame_chunks": [{"key": "frames_000"}, {"key": "frames_999"}]}  # 999 missing on purpose
    json.dump(md, open(f"{folder}/match_data.json", "w"))
    json.dump({"frames": frames_chunk}, open(f"{folder}/frames_000.json", "w"))
    if summary:
        json.dump({"possession_pct": [55, 45], "passes": 12, "turnovers": 3, "ball_grade": 0.8}, open(f"{folder}/summary.json", "w"))


def _keys(ref):
    os.makedirs(ref, exist_ok=True)
    # frames at fps 10: 10 -> t 1.0, 20 -> 2.0, 30 -> 3.0 (bridged ball), 40 -> 4.0 (no frame within 0.2 s)
    json.dump({"10": [100, 100], "20": [200, 200], "30": [300, 300], "40": [0, 0]}, open(f"{ref}/ball_key_px.json", "w"))
    json.dump({"items": [{"t": 1.0, "owner": "A"}, {"t": 2.0, "owner": "?"}]}, open(f"{ref}/ball_key_graded.json", "w"))
    json.dump({"items": [{"t": 2.05, "owner": "B"}, {"t": 3.0, "owner": "B"}, {"t": 9.0, "owner": "A"}]}, open(f"{ref}/poss_sample.json", "w"))


FR = [{"t": 1.0, "ball": {"px": [110, 100], "state": "seen"}, "possession": "A"},     # 10 px: hit
      {"t": 3.0, "ball": {"px": [300, 300], "state": "bridged"}, "possession": "A"}]  # bridged: not counted; owner B wrong
CH = [{"t": 2.0, "ball": {"px": [245, 200], "state": "seen"}, "possession": "B"}]     # 45 px: hit60 only; owner B right


def test_grade_counts(tmp_path):
    folder, ref = str(tmp_path / "exp" / "m1"), str(tmp_path / "ref")
    _export(folder, FR, CH); _keys(ref)
    out = Q.grade(folder, ref, log=lambda *_: None)
    assert out["n_frames"] == 3
    assert out["ball"] == {"hit30": 1, "hit60": 2, "seen": 2, "n": 4}
    # '?' dropped; t 9.0 has no frame within 0.2 s -> skipped; A at 1.0 right, B at 2.05 right, B at 3.0 wrong
    assert out["owner"] == {"agree": 2, "n": 3, "by_team": {"A": [1, 1], "B": [1, 2]}}
    assert out["possession_pct"] == [55, 45] and out["summary"]["passes"] == 12


def test_missing_keys_and_empty_export(tmp_path):
    folder = str(tmp_path / "m2")
    _export(folder, [], [], summary=False)
    out = Q.grade(folder, str(tmp_path / "noref"), log=lambda *_: None)
    assert out == {"n_frames": 0}
    ref = str(tmp_path / "ref"); _keys(ref)
    out = Q.grade(folder, ref, log=lambda *_: None)  # keys but no frames: nothing seen, no crash
    assert out["ball"]["seen"] == 0 and out["owner"]["n"] == 0


def test_cli_finds_export_layout(tmp_path):
    exp = tmp_path / "full_match_x" / "export"
    _export(str(exp / "nokey-match"), FR, CH)
    dst = tmp_path / "full_match_x" / "qa.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "export_qa.py"), str(exp), "", str(dst)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.load(open(dst))
    assert out["match"] == "nokey-match" and out["n_frames"] == 3 and "ball" not in out


def test_real_vallentuna_copy():
    folder = os.path.join(ROOT, "results/volume/runs/matches/p15u-vs-vallentuna-2026-10-03-6cce")
    if not os.path.exists(f"{folder}/match_data.json"):
        pytest.skip("repo copy of the Vallentuna export missing")
    out = Q.grade(folder, os.path.join(ROOT, "reference/p15u-vs-vallentuna-2026-10-03-6cce"), log=lambda *_: None)
    assert out["ball"]["n"] == 55 and out["owner"]["n"] == 39
    assert out["ball"]["hit30"] >= 40  # 43/55 on the 4 Oct export
