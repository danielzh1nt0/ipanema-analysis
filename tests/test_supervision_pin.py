"""18 Oct (worker, N0): supervision 0.31 removed ByteTrack, so every nightly player check failed on 17 and 18 Oct
(AttributeError in ipanema/tracking.py) while the card still said 'nothing worse' (it showed old numbers).
Guards: every pip install of supervision in the workflows / Kaggle / tools is capped below 0.31; tracking says plainly
what is wrong if a too-new supervision is installed anyway; a failed player job shows on the card as NOT RUN."""
import glob, os, re, sys, types
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_lines():
    files = glob.glob(f"{ROOT}/.github/workflows/*.yml") + glob.glob(f"{ROOT}/kaggle/*.py") + glob.glob(f"{ROOT}/tools/*.py")
    for f in files:
        for i, ln in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
            if "pip install" in ln and re.search(r"\bsupervision\b", ln): yield os.path.relpath(f, ROOT), i, ln


def test_every_supervision_install_is_capped_below_031():
    bad = []
    for f, i, ln in _install_lines():
        for spec in re.findall(r"""supervision[^\s"']*""", ln):
            if not re.search(r"<\s*0\.31|<\s*0\.30|<\s*0\.2\d|==\s*0\.(2\d|30)", spec): bad.append(f"{f}:{i}: {spec}")
    assert not bad, "supervision without a cap below 0.31:\n" + "\n".join(bad)


def test_caps_are_quoted_for_the_shell():
    # an unquoted '<0.31' in a shell command is a redirect from a file called 0.31, not a version cap
    bad = [f"{f}:{i}" for f, i, ln in _install_lines() if re.search(r"""(?<!["'])supervision[>=<][^\s"']*<""", ln)]
    assert not bad, bad


def test_requirements_cap():
    assert re.search(r"^supervision<0\.(2\d|3[01])\b", open(f"{ROOT}/requirements.txt").read(), re.M)


def test_bytetrack_plain_error_on_new_supervision(monkeypatch):
    from ipanema import tracking as TR
    fake = types.ModuleType("supervision"); fake.__version__ = "0.31.0"
    monkeypatch.setitem(sys.modules, "supervision", fake)
    with pytest.raises(RuntimeError, match=r"0\.31\.0 has no ByteTrack.*<0\.31"):
        TR.bytetrack(25)


def test_bytetrack_on_pinned_supervision():
    sv = pytest.importorskip("supervision")
    if not hasattr(sv, "ByteTrack"): pytest.skip("installed supervision is too new (the pin guards the runners)")
    from ipanema import tracking as TR
    assert TR.bytetrack(25.0).__class__.__name__ == "ByteTrack"


# ---- the card ----
from tools import nightly_scorecard as N

Q = {"tests": {"total": 10, "failed": []}, "ball_34_moments": {"right": 22, "of": 34},
     "stats_pro_data": {"worst_pass_err_pct": 15, "worst_poss_err_pts": 8}}
S = {"new, RF-DETR": {"observed_per_frame": {"dark": 7.0, "light": 6.0}, "median_track_s": {"dark": 2.4, "light": 2.5}}}
SP = "results/qa/tracktest_p15u-vs-spanga-2026-09-25/summary.json"
SFK = "results/qa/tracktest/summary.json"
GPU = "results/qa/tracktest_gpu/summary.json"


def test_nightly_folders_from_the_real_workflow():
    got = N.nightly_folders(f"{ROOT}/.github/workflows/nightly.yml")
    assert "tracktest" in got and "tracktest_p15u-vs-spanga-2026-09-25" in got and "tracktest_gpu" not in got


def test_failed_jobs_marked_not_run(tmp_path):
    art = tmp_path / "art"; (art / "quick").mkdir(parents=True)
    (art / "players-SFKBP1109_s1200" / "tracktest_p15u-vs-spanga-2026-09-25").mkdir(parents=True)   # every artifact holds all folders
    yml = f"{ROOT}/.github/workflows/nightly.yml"
    summ = [(SFK, S), (SP, S), (GPU, S)]
    stale = N.stale_summaries(summ, art=str(art), yml=yml)
    assert stale == {SP}                          # spanga's job uploaded nothing; SFK delivered; gpu is not a nightly job
    lines = N.build_lines("2026-10-18", Q, summ, stale)
    assert any(l.startswith("| Players, p15u-vs-spanga") and "NOT RUN" in l for l in lines)
    assert any(l.startswith("| Players, SFKBP1109 | seen/frame") for l in lines)
    prev = N.read_card(N.build_lines("2026-10-17", Q, summ))
    w = N.worse_than(prev, N.read_card(lines))
    assert w == ["p15u-vs-spanga-2026-09-25: player check did not run (job failed)"]


def test_all_failed_like_17_and_18_oct(tmp_path):
    art = tmp_path / "art"; (art / "quick").mkdir(parents=True)                # only the quick job delivered
    summ = [(SFK, S), (SP, S)]
    stale = N.stale_summaries(summ, art=str(art), yml=f"{ROOT}/.github/workflows/nightly.yml")
    assert stale == {SFK, SP}
    w = N.worse_than(N.read_card(N.build_lines("a", Q, summ)), N.read_card(N.build_lines("b", Q, summ, stale)))
    assert len(w) == 2 and all("did not run" in x for x in w)


def test_local_run_without_artifacts_marks_nothing(tmp_path):
    assert N.stale_summaries([(SFK, S)], art=str(tmp_path / "none")) == set()
