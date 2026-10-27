"""N0-27c: a nightly whose scorecard job never starts still gets its card (tools/nightly_rescue.py)."""
import io, json, os, shutil, sys, urllib.error, zipfile
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import nightly_rescue as R
from tools import nightly_scorecard as N


def job(name, status="completed", conclusion="success"):
    return {"name": name, "status": status, "conclusion": conclusion if status == "completed" else None}


# the real 27 Oct run 55: 6 checks done, scorecard queued for 16 h
RUN55 = [job("players (p15u-vs-spanga-2026-09-25, 1500)"), job("players (SFKBP1109_s1200, 60)"), job("quick"),
         job("players (p15u-vs-djursholm-2026-09-26, 1500)"), job("scorecard", "queued")]


def test_stuck_scorecard_needs_rescue():
    assert R.needs_rescue(RUN55)


def test_normal_and_running_runs_left_alone():
    assert not R.needs_rescue(RUN55[:-1] + [job("scorecard")])                 # card written
    assert not R.needs_rescue([job("quick", "in_progress"), job("scorecard", "queued")])   # checks still running
    assert not R.needs_rescue([])                                              # pending run, no jobs yet
    assert R.needs_rescue(RUN55[:-1] + [job("scorecard", conclusion="failure")])   # card job failed


def test_pick_newest_stuck_run():
    runs = [{"id": 56}, {"id": 55}, {"id": 54}]
    jobs = {56: [], 55: RUN55, 54: RUN55[:-1] + [job("scorecard")]}
    assert R.pick_run(runs, jobs.get)["id"] == 55
    assert R.pick_run([{"id": 54}], jobs.get) is None


def _zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, s in files.items(): z.writestr(n, s)
    return b.getvalue()


def test_extract_refuses_paths_outside(tmp_path):
    with pytest.raises(ValueError): R.extract(_zip({"../evil.py": "x"}), str(tmp_path / "a"))
    R.extract(_zip({"quick.json": "{}"}), str(tmp_path / "q"))
    assert (tmp_path / "q" / "quick.json").exists()


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_download_follows_redirect_without_token():
    seen = {}

    def api(req):
        seen["auth"] = req.headers.get("Authorization")
        raise urllib.error.HTTPError(req.full_url, 302, "Found", {"Location": "https://blob/x.zip?sig=1"}, None)

    def blob(url):
        seen["blob"] = url; return _Resp(b"ZIP")
    assert R.download("https://api/zip", "tok", open_api=api, open_blob=blob) == b"ZIP"
    assert seen == {"auth": "Bearer tok", "blob": "https://blob/x.zip?sig=1"}       # blob got a bare url, no token


def _summary(dark, light):
    return json.dumps({"new, RF-DETR": {"observed_per_frame": {"dark": dark, "light": light}, "median_track_s": {"dark": 2.44, "light": 2.54}}})


def test_dry_run_builds_dated_card_like_the_scorecard_job(tmp_path, monkeypatch):
    """artifacts laid out as GitHub delivers them -> place() -> scorecard under the run's date, compared with the day before"""
    art = tmp_path / "art"
    R.extract(_zip({"quick.json": json.dumps({"tests": {"total": 10, "failed": []}, "ball_34_moments": {"right": 22, "of": 34, "best_possible": 25},
                                               "stats_pro_data": {"worst_pass_err_pct": 15, "worst_poss_err_pts": 8}})}), str(art / "quick"))
    R.extract(_zip({"tracktest_p15u-vs-spanga-2026-09-25/summary.json": _summary(7.0, 5.0)}), str(art / "players-p15u-vs-spanga-2026-09-25"))
    root = tmp_path / "repo"; (root / "results" / "nightly").mkdir(parents=True)
    (root / "results" / "nightly" / "2026-10-26.md").write_text(
        "| Ball, 34 checked moments | 22/34 (best possible 25) |\n| Players, p15u-vs-spanga-2026-09-25 | seen/frame dark 7.0 light 6.0, tracked 2.44/2.54 s |\n")
    R.place(str(art), str(root))
    assert (root / "results" / "qa" / "tracktest_p15u-vs-spanga-2026-09-25" / "summary.json").exists()
    monkeypatch.chdir(root); monkeypatch.setenv("NIGHTLY_DATE", "2026-10-27")
    monkeypatch.setattr(N, "ART", str(art))
    N.main()
    card = (root / "results" / "nightly" / "2026-10-27.md").read_text()
    assert "# Nightly check 2026-10-27" in card and "| Tests | 10/10 pass |" in card
    assert "| Worse than 2026-10-26 | p15u-vs-spanga-2026-09-25: light players seen per frame 6 -> 5 |" in card
