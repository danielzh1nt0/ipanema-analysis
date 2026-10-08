"""C3b (8 Oct): the blind sample of Vallentuna camera rows - group rules, no group leaks into the drawn sample, repeatable."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import c3b_sample as CS
P = [0.1, 0.2, 0.0, 1200.0]

def test_groups():
    assert CS.group({"pose": P, "fwd_bwd_px": 5.0, "why": []}) == "brave"
    assert CS.group({"pose": P, "fwd_bwd_px": 20.0, "why": ["forward/backward disagree"]}) == "fb15-25"
    assert CS.group({"pose": P, "fwd_bwd_px": 30.0, "why": ["refine doubtful", "forward/backward disagree"]}) == "fb25-40"
    assert CS.group({"pose": P, "fwd_bwd_px": 300.0, "why": ["forward/backward disagree"]}) == "fb100+"
    assert CS.group({"pose": P, "fwd_bwd_px": 20.0, "why": ["cold and tracked disagree", "forward/backward disagree"]}) is None
    assert CS.group({"pose": P, "anchor": True, "why": ["lines and paint disagree (9 px)", "jump from track"]}) == "jump_anchor"
    assert CS.group({"pose": None, "why": []}) is None

def test_pick_repeatable_and_in_play():
    rows = [{"t": float(t), "pose": P, "fwd_bwd_px": 20.0 if t % 2 else 3.0, "why": ["forward/backward disagree"] if t % 2 else []} for t in range(0, 400)]
    a, b = CS.pick(rows, [[100, 300]]), CS.pick(rows, [[100, 300]])
    assert [q["t"] for q in a] == [q["t"] for q in b]
    assert all(105 <= q["t"] <= 295 for q in a)
    assert sum(q["group"] == "fb15-25" for q in a) == 20 and sum(q["group"] == "brave" for q in a) == 8

def test_committed_sample_has_no_group():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/qa/c3b/sample.json")
    if os.path.exists(p):
        assert all(set(r) == {"id", "t", "pose"} for r in json.load(open(p))["rows"])
