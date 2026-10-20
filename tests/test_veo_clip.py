"""20 Oct (N0-20a): a clip cut from a match ('<match>_s<start>[_d<dur>]') reads its match's Veo shots/goals, shifted to the
clip's clock. Before, it looked for reference/veo_highlights_<clip>.txt, found none and showed NO shots or goals: the app's
AIK clip (from 2520 s) lost 3 Veo shots and AIK's first goal (2762 s, B by eye)."""
import json, os
import numpy as np
from ipanema import veo as VEO
from ipanema.run import veo_source, clip_of, clip_period_index, periods_direction

AIK = "p15u-vs-aik-2026-09-21-bd09"

def test_clip_of():
    assert clip_of("SFKBP1109_s1200") == ("SFKBP1109", 1200.0, None)
    assert clip_of(f"{AIK}_s2520") == (AIK, 2520.0, None)
    assert clip_of("m_s4200_d300") == ("m", 4200.0, 300.0)
    assert clip_of("SFKBP1109") is None and clip_of(AIK) is None

def test_aik_clip_gets_its_shots_and_goal():
    hl, tm, og = veo_source("/nonexistent", f"{AIK}_s2520", 300.0, 106.0, 64.0, log=lambda *a: None)
    assert hl == [(12.0, "shot"), (95.0, "shot"), (241.0, "shot"), (242.0, "goal")]
    assert tm == {242.0: "B"} and og == {}
    # through build: the goal swallows its own shot clip (within 2 s) -> 2 shots + AIK's goal, team by eye
    out, rej = VEO.build(hl, 30.0, {}, {}, 106.0, 64.0, {"A": True, "B": False}, [], log=lambda *a: None, known_teams=tm, known_origins=og)
    assert rej == [] and [(s["t"], s["goal"]) for s in out] == [(12.0, False), (95.0, False), (242.0, True)]
    assert out[2]["team"] == "B" and out[2]["located_by"] == "by eye"

def test_sfkbp_clip_has_none_inside_and_matches_unchanged():
    assert veo_source("/nonexistent", "SFKBP1109_s1200", 300.0, 106.0, 64.0, log=lambda *a: None) == ([], {}, {})   # 863 and 2183 s are outside
    hl, tm, _ = veo_source("/nonexistent", "SFKBP1109", 3300.0, 106.0, 64.0)                                         # a full match: the file as it is
    assert (2619, "goal") in hl and tm[2619] == "A" and len(hl) == 29
    assert veo_source("/nonexistent", "no-such-match_s100", 300.0, 106.0, 64.0) is None

def test_window_shift_and_mirror(tmp_path):
    (tmp_path / "reference").mkdir(); (tmp_path / "periods").mkdir()
    open(tmp_path / "reference" / "veo_highlights_m.txt", "w").write(
        "# test\n50 shot\n120 shot A 90.0 20.0\n121 goal A 95.5 30.0\n400 shot\n1500 shot B 10.0 40.0\n1700 shot\n")
    json.dump({"periods_s": [[0, 1000], [1200, 2200]], "attack_right_A": True}, open(tmp_path / "periods" / "m.json", "w"))
    r = str(tmp_path); q = lambda *a: None
    hl, tm, og = veo_source(r, "m_s100", 300.0, 106.0, 64.0, log=q)            # first half: spots as in the file
    assert hl == [(20.0, "shot"), (21.0, "goal"), (300.0, "shot")] and tm == {20.0: "A", 21.0: "A"} and og == {20.0: (90.0, 20.0), 21.0: (95.5, 30.0)}
    hl, tm, og = veo_source(r, "m_s1400", 300.0, 106.0, 64.0, log=q)           # second half (an event at the very end is kept, like veo_in_window): spots mirrored back to the clip's own view
    assert hl == [(100.0, "shot"), (300.0, "shot")] and tm == {100.0: "B"} and og == {100.0: (96.0, 24.0)}
    assert clip_period_index(r, "m_s1400") == 1 and periods_direction(r, "m_s1400") is False
    hl, _, _ = veo_source(r, "m_s1400", 200.0, 106.0, 64.0, log=q)             # the real clip length decides the end, not 300 s
    assert hl == [(100.0, "shot")]
    # the match's file wins over a clip-only file without letters (what modal_app.prepare_match_files writes)
    open(tmp_path / "reference" / "veo_highlights_m_s100.txt", "w").write("20 shot\n21 goal\n")
    assert veo_source(r, "m_s100", 300.0, 106.0, 64.0, log=q)[1] == {20.0: "A", 21.0: "A"}
    os.remove(tmp_path / "reference" / "veo_highlights_m.txt")                 # no match file: the clip's own file is used
    assert veo_source(r, "m_s100", 300.0, 106.0, 64.0, log=q) == ([(20, "shot"), (21, "goal")], {}, {})

def test_shift_no_end():
    hl, tm, og = VEO.shift([(5, "shot"), (15, "goal")], {15: "B"}, {15: (1.0, 2.0)}, 10.0, None, True, 106.0, 64.0)
    assert hl == [(5.0, "goal")] and tm == {5.0: "B"} and og == {5.0: (105.0, 62.0)}
