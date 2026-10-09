"""S2 (9 Oct): scoring motion vs ball restarts against Veo minutes on the full SFK-BP half (tools/s2full.py)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import s2full as S
from ipanema import possession as P


def test_veo_truth_minutes_map_to_video_seconds(tmp_path):
    f = tmp_path / "veo.txt"; f.write_text("# head\n12 SFK Throw-in\n12 BP1 Shot\n35 SFK Corner\n44 BP1 Goal kick\n")
    T = S.veo_truth(str(f), cover=[(11, 31), (44, 45)])
    assert [(t["minute"], t["kind"], t["t0"]) for t in T] == [(12, "throw-in", 660.0), (44, "goal kick", 2580.0)]


def test_match_is_one_to_one():
    T = [{"minute": 12, "kind": "throw-in", "t0": 660.0, "t1": 720.0}, {"minute": 12, "kind": "corner", "t0": 660.0, "t1": 720.0}]
    found, used = S.match([700.0], T)
    assert sum(f[2] is not None for f in found) == 1 and used == {0}
    found, used = S.match([700.0, 710.0, 800.0], T)
    assert sum(f[2] is not None for f in found) == 2


def test_union_counts_near_ones_once():
    assert S.union([100.0, 200.0], [104.0, 300.0]) == [100.0, 200.0, 300.0]


def test_score_extras_only_in_cover():
    T = [{"minute": 12, "kind": "throw-in", "t0": 660.0, "t1": 720.0}]
    s = S.score([700.0, 900.0, 2000.0], T, [(11, 16)])         # 2000 s is outside the covered minutes
    assert s["ours"] == 2 and s["found"] == 1 and s["extras"] == 1 and s["extra_t"] == [900.0]


def test_per_from_frames_skips_uncalibrated_and_off_team():
    fr = [{"t": 0.0, "players": [{"id": 1, "team": "A", "m": [10, 10]}, {"id": 2, "team": None, "m": [5, 5]}, {"id": 3, "team": "B", "m": None}]}]
    per = S.per_from_frames(fr)
    assert len(per[0]) == 1 and per[0][0][0] == 1


def test_motion_finds_a_standstill_in_exported_rows():
    fps = 10.0; fr = []
    for k in range(300):                                        # 30 s: run, stand 8 s (k 100-180), run
        v = 0.0 if 100 <= k < 180 else 0.6                       # metres per frame = 6 m/s
        fr.append({"t": k / fps, "players": [{"id": i, "team": "AB"[i % 2], "m": [5 + i * 3 + v * k, 30.0]} for i in range(8)]})
    st = P.stoppages_from_motion(S.per_from_frames(fr), fps, thr=1.4, min_s=3.0)
    assert len(st) == 1 and 95 <= st[0][0] <= 115 and 170 <= st[0][1] <= 190
