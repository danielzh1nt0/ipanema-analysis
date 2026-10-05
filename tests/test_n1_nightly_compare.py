"""N1: the nightly card compares itself with the day before."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import nightly_scorecard as N

Q = {"tests": {"total": 100, "failed": []}, "ball_34_moments": {"right": 22, "of": 34, "best_possible": 25},
     "stats_pro_data": {"worst_pass_err_pct": 15, "worst_poss_err_pts": 8, "sequences_found_pct": 40, "sequences_real_pct": 44}}


def summ(dark, light, td=2.6, tl=3.19):
    return {"new, RF-DETR": {"observed_per_frame": {"dark": dark, "light": light}, "median_track_s": {"dark": td, "light": tl}}}


F = "results/qa/tracktest_solberga-vs-p09-norrviken-2026-09-11/summary.json"


def test_same_numbers_nothing_worse():
    a = N.build_lines("2026-10-04", Q, [(F, summ(7.0, 7.0))])
    assert N.worse_than(N.read_card(a), N.read_card(a)) == []


def test_solberga_light_drop_found():
    # the real 4 -> 5 Oct change (P2e turned the screen-edge keeper rule off; near white #5 dropped)
    a = N.build_lines("2026-10-04", Q, [(F, summ(7.0, 7.0))])
    b = N.build_lines("2026-10-05", Q, [(F, summ(7.0, 6.0))])
    w = N.worse_than(N.read_card(a), N.read_card(b))
    assert w == ["solberga-vs-p09-norrviken-2026: light players seen per frame 7 -> 6"]


def test_stats_and_tests_worse_and_better():
    q2 = {**Q, "tests": {"total": 100, "failed": ["x"]}, "ball_34_moments": {"right": 20, "of": 34},
          "stats_pro_data": {"worst_pass_err_pct": 18, "worst_poss_err_pts": 7, "sequences_found_pct": 41, "sequences_real_pct": 44}}
    w = N.worse_than(N.read_card(N.build_lines("d", Q, [])), N.read_card(N.build_lines("d", q2, [])))
    assert "tests failing (1)" in w and "ball moments right 22 -> 20" in w and "pass count error % 15 -> 18" in w
    assert not any("possession" in x or "sequences" in x for x in w)   # better or within noise


def test_tracked_time_drop_needs_20pct():
    a = N.read_card(N.build_lines("d", Q, [(F, summ(7, 7, 2.6, 3.19))]))
    assert N.worse_than(a, N.read_card(N.build_lines("d", Q, [(F, summ(7, 7, 2.2, 3.19))]))) == []
    assert N.worse_than(a, N.read_card(N.build_lines("d", Q, [(F, summ(7, 7, 1.9, 3.19))])))[0].endswith("dark median time tracked 2.6 -> 1.9 s")


def test_compare_row_uses_latest_earlier_card(tmp_path):
    old = N.build_lines("2026-10-03", Q, [(F, summ(7.0, 8.0))]); prev = N.build_lines("2026-10-04", Q, [(F, summ(7.0, 7.0))])
    (tmp_path / "2026-10-03.md").write_text("\n".join(old)); (tmp_path / "2026-10-04.md").write_text("\n".join(prev))
    (tmp_path / "2026-10-05.md").write_text("\n".join(prev))   # today's earlier run must not count
    new = N.build_lines("2026-10-05", Q, [(F, summ(7.0, 6.0))])
    assert N.compare_row("2026-10-05", new, str(tmp_path)) == "| Worse than 2026-10-04 | solberga-vs-p09-norrviken-2026: light players seen per frame 7 -> 6 |"
    assert N.compare_row("2026-10-03", new, str(tmp_path)) is None


def test_real_cards_read():
    # the committed 4 and 5 Oct cards parse and show the Solberga drop
    p4, p5 = os.path.join(ROOT, "results/nightly/2026-10-04.md"), os.path.join(ROOT, "results/nightly/2026-10-05.md")
    if not (os.path.exists(p4) and os.path.exists(p5)): return
    w = N.worse_than(N.read_card(open(p4).read().splitlines()), N.read_card(open(p5).read().splitlines()))
    assert w == ["solberga-vs-p09-norrviken-2026: light players seen per frame 7 -> 6"]
