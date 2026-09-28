"""E2: the third who-has-the-ball batch is chosen blind and never overlaps an earlier strip (+-0.8 s)."""
import json, os
from tools.who_moments3 import pick_frames

def test_pick_frames_keeps_gap():
    taken = [100, 500, 900]
    fr = pick_frames(1200, taken, gap=36, step=23)
    assert fr and all(abs(a - b) >= 36 for a in fr for b in taken)
    assert all(abs(a - b) >= 36 for i, a in enumerate(fr) for b in fr[i + 1:])
    assert min(fr) >= 30 and max(fr) < 1170

def test_committed_batch_does_not_overlap_earlier_keys():
    p = "results/review/who_moments3.json"
    if not os.path.exists(p): return
    new = [m["frame"] for m in json.load(open(p))["moments"]]
    old = [m["frame"] for f in ("who_moments.json", "who_moments2.json") for m in json.load(open("results/review/" + f))["moments"]]
    assert all(abs(a - b) >= 36 for a in new for b in old)

def test_answer_key_third_batch_matches_strips():
    A = json.load(open("results/review/who_answers.json"))["moments"]
    third = [a for a in A if a["n"] >= 200]
    M = json.load(open("results/review/who_moments3.json"))["moments"]
    assert [a["frame"] for a in third] == [m["frame"] for m in M]
    assert all(a["truth"] in ("dark", "white", "loose", "unsure") for a in A)
    assert sum(a["truth"] != "unsure" for a in A) >= 99
    assert len({a["n"] for a in A}) == len(A)
