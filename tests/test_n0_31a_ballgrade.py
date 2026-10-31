"""N0-31a: the ball grade (summary.ball_grade -> possession_ok / events_ok) uses checked frames, not the near-a-player test."""
import os, sys, json, subprocess, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from ipanema import run as R  # noqa: E402
import n0_31a_ballgrade as G  # noqa: E402


def test_old_rule_matches_the_lab_copy():
    for acc, nf, fp in itertools.product((None, 0.5, 0.6, 0.82), (0.47, 0.79, 0.85, 1.0), (0.3, 1.0)):
        assert R.ball_grade(acc, nf, fp, mode="old") == G.gate(acc, nf, fp)


def test_new_rule_today_matches():
    # the repo's app copies: SFK-BP acc 0.82 near 0.47, AIK 0.67 / 0.50, Vallentuna no clicks / 0.62 -> unchanged possession verdicts
    assert R.ball_grade(0.82, 0.47, 1.0, mode="checked") == (True, False)
    assert R.ball_grade(0.67, 0.50, 1.0, mode="checked") == (True, False)
    assert R.ball_grade(None, 0.62, 1.0, mode="checked") == (False, False)


def test_new_rule_ignores_near_a_player():
    # a ball parked on a random player (near 1.0) with no clicks was 'possession OK' under the old rule
    assert R.ball_grade(None, 1.0, 1.0, mode="old")[0] is True
    assert R.ball_grade(None, 1.0, 1.0, mode="checked") == (False, False)
    # a perfect ball seen through our camera (near ~0.6) can now get events OK - once a by-eye event check passed
    assert R.ball_grade(0.95, 0.6, 1.0, mode="old")[1] is False
    assert R.ball_grade(0.95, 0.6, 1.0, events_checked=True, mode="checked") == (True, True)
    assert R.ball_grade(0.95, 0.6, 1.0, events_checked=False, mode="checked") == (True, False)
    assert R.ball_grade(0.5, 0.95, 1.0, events_checked=True, mode="checked") == (False, False)


def test_events_check_file(tmp_path):
    (tmp_path / "overrides").mkdir()
    assert R.events_checked(str(tmp_path), "m1") is False
    (tmp_path / "overrides" / "m1_events_check.json").write_text(json.dumps({"passed": True}))
    (tmp_path / "overrides" / "m2_events_check.json").write_text("not json")
    assert R.events_checked(str(tmp_path), "m1") is True and R.events_checked(str(tmp_path), "m2") is False


def test_lab_dry_run():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "n0_31a_ballgrade.py"), "--dry"], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0 and "dry ok" in r.stdout, r.stderr


def test_metrica_result_if_present():
    f = os.path.join(ROOT, "results", "qa", "n0-31a", "metrica_ballgrade.json")
    if not os.path.exists(f): return
    rows = json.load(open(f))["rows"]
    perfect = [r for r in rows if r["ball"] == "perfect"]; sticky = [r for r in rows if r["ball"] == "sticky"]
    assert all(r["near_frac"] < 0.9 for r in perfect)                       # the old 0.85 bar is near the ceiling of a PERFECT ball
    assert all(r["near_frac"] > 0.95 and r["possession_ok_no_check"] for r in sticky)   # and a coin-flip ball passed it
    assert all(not r["possession_ok"] for r in rows if r["ball"] in ("garbage", "sticky"))   # the checked accuracy catches both
    assert all(r["possession_ok"] for r in rows if r["ball"] in ("perfect", "flicker", "streak"))   # veo 6/8: 40 clicks at 2 m are noisy
