"""A1: the paid-run plan is checked before Daniel sees it (no Modal involved)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import a1_paidrun as A

ROOT = os.path.join(os.path.dirname(__file__), "..")
WF = open(os.path.join(ROOT, ".github/workflows/run-on-modal.yml")).read()

def test_real_plan_is_clean():
    assert A.check(A.PLAN, WF, ROOT) == []

def test_unknown_tag_and_stray_run_caught():
    s = dict(A.PLAN[0], msg="x [fullx:SFKBP1109]", match="SFKBP1109")
    bad = A.check([s], WF, ROOT)
    assert any("no Modal job tag" in b for b in bad)
    s2 = dict(A.PLAN[0], msg="x [full:SFKBP1109] [budget:30] see [run notes]", match="SFKBP1109")
    assert any("'[run'" in b for b in A.check([s2], WF, ROOT))

def test_join_needs_budget():
    s = dict(A.PLAN[2], msg="x [full:SFKBP1109] [allow-fallback]")
    assert any("budget" in b for b in A.check([s], WF, ROOT))

def test_retrack_with_override_in_place_is_refused():
    s = dict(A.PLAN[-1]); s.pop("override_aside")
    assert any("override" in b for b in A.check([s], WF, ROOT, overrides_present=True))
    assert A.check([A.PLAN[-1]], WF, ROOT, overrides_present=True) == []
    assert "git mv overrides/" in A.push_cmd(A.PLAN[-1])

def test_costs_positive_and_gpu_dearer():
    c = {s["id"]: A.cost_eur(s) for s in A.PLAN}
    assert all(v >= 0 for v in c.values()) and c["1"] > 0 and c["6"] > c["1"] > c["2"]

def test_push_touches_a_watched_path():
    for s in A.PLAN:
        assert "triggers/last.txt" in A.push_cmd(s) and s["msg"] in A.push_cmd(s)
