"""N0-30a: the blind AIK possession sample and its comparison tool."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import n0_30a_aik_poss as T

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _fr(seq, t0=0.0, dt=0.3):
    return [{"t": round(t0 + i * dt, 3), "possession": p, "phase": "control" if p else "loose"} for i, p in enumerate(seq)]


def test_nearest_and_window_majority():
    fr = _fr(["A", "A", "B", "A", "A", "B", "B"])
    ts = [f["t"] for f in fr]
    assert T.nearest(ts, 0.62) == 2                     # 0.6 is the closest row
    assert T.window_owner(fr, ts, 0.6, half=1.0) == "A"  # 4 A vs 3 B in 0..1.6 s; one flicker row does not decide
    assert T.window_owner(_fr(["A", "B"]), [0.0, 0.3], 0.15) is None   # tie -> undecided


def test_analyse_counts_and_agreement():
    fr = _fr(["A"] * 10 + ["B"] * 10)                    # A for 0-2.7 s, B for 3.0-5.7 s
    g = [{"file": "p0.jpg", "t": 0.9, "eye": "A", "sure": True, "dead": False, "note": ""},
         {"file": "p1.jpg", "t": 4.5, "eye": "A", "sure": False, "dead": False, "note": ""},   # wrong on purpose
         {"file": "p2.jpg", "t": 5.1, "eye": "B", "sure": True, "dead": True, "note": ""},
         {"file": "p3.jpg", "t": 1.5, "eye": None, "sure": False, "dead": False, "note": ""}]
    res, items = T.analyse(fr, g, half=(0, 6))
    a = res["all_decided"]
    assert (a["eye_sfk"], a["eye_decided"]) == (2, 3) and a["agree"] == 2 and a["agree_of"] == 3
    assert res["sure_only"]["eye_decided"] == 2 and res["sure_only"]["agree"] == 2
    assert res["in_play_only"]["eye_decided"] == 2
    assert res["eye_undecided"] == 1 and res["export_first_half_sfk_pct"] == 50.0


def test_wilson_interval_brackets_the_share():
    lo, hi = T.wilson(8, 27)
    assert lo < 30 < hi and (lo, hi) == (16, 48)
    assert T.wilson(0, 0) == (None, None)


def test_committed_grades_are_complete_and_blind():
    G = json.load(open(os.path.join(ROOT, "results/qa/n0-30a/aik_poss_grades.json")))
    items = G["items"]
    assert len(items) == 60 and len({x["file"] for x in items}) == 60
    assert all(x["eye"] in ("A", "B", None) for x in items)
    assert not any("ours" in x for x in items)          # grades file holds no pipeline answer
    times = {f["file"]: f["t"] for f in json.load(open(os.path.join(ROOT, "results/kaggle/aik_poss_sample/poss/times.json")))["frames"]}
    assert all(abs(times[x["file"]] - x["t"]) < 1e-6 for x in items)
    k, n = T.share(items)
    assert (k, n) == (8, 27)
