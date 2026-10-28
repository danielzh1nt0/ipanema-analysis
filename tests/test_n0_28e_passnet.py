"""N0-28e: the pass-network / pass-length check on Metrica (tools/n0_28e_passnet.py) measures what the export writes."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from ipanema import metrics as MT
import n0_28e_passnet as NP


def _p(t, a, b, team="A", done=True, fm=(10, 10), tm=(20, 10), kind="forward"):
    return {"t": t, "from": a, "to": b, "team": team, "completed": done, "from_m": list(fm), "to_m": list(tm), "kind": kind, "progressive": False,
            "length_m": float(np.hypot(tm[0] - fm[0], tm[1] - fm[1]))}


PS = [_p(1, 1, 2), _p(2, 1, 2, fm=(12, 10)), _p(3, 2, 1, fm=(20, 12), tm=(10, 12)), _p(4, 2, 3, done=False), _p(5, 7, 8, team="B")]


def test_network_counts_completed_only_and_nodes():
    n = NP.network(PS); e, nodes = n["A"]
    assert e == {(1, 2): 2, (2, 1): 1}                                         # the lost pass 2->3 is not an edge
    assert np.allclose(nodes[1], [(10 + 12 + 10) / 3, (10 + 10 + 12) / 3])     # 2 passes played + 1 received
    assert n["B"][0] == {(7, 8): 1}


def test_lab_reads_exactly_what_the_export_writes():
    net = MT.pass_network(PS, NP.players_stub(PS), nodes="passes"); ex = NP.from_export(net); own = NP.network(PS)
    for tm in ("A", "B"):
        assert ex[tm][0] == own[tm][0]
        for i in own[tm][1]: assert np.allclose(ex[tm][1][i], own[tm][1][i], atol=0.06)   # export rounds to 0.1 m


def test_compare_identical_and_disjoint():
    e = {(1, 2): 5, (2, 3): 3, (3, 1): 1}; nd = {1: np.zeros(2), 2: np.ones(2), 3: np.ones(2) * 2}
    c = NP.compare((e, nd), (dict(e), dict(nd)))
    assert c["weighted_overlap"] == 1.0 and c["top10_overlap"] == 3 and c["busiest_right"] and c["node_err_median_m"] == 0.0 and c["spearman"] == 1.0
    d = NP.compare((e, nd), ({(5, 6): 4}, {5: np.zeros(2)}))
    assert d["weighted_overlap"] == 0.0 and d["top10_overlap"] == 0 and not d["busiest_right"] and d["nodes_both"] == 0


def test_volume_null_keeps_player_totals_and_scores_below_a_real_pattern():
    e = {(1, 2): 10, (2, 1): 10, (3, 4): 10, (4, 3): 10}                       # two closed pairs: strong pair structure
    nl = NP.volume_null(e)
    assert abs(sum(v for (i, j), v in nl.items() if i == 1) - 10 * (1 - 10 / 40)) < 1e-9   # self-pass removed
    assert NP.overlap(e, nl) < 0.5 and NP.overlap(e, e) == 1.0


def test_match_lengths_pairs_same_team_completed_within_1s():
    T = [{"team": "A", "f0": 25, "from": 1, "to": 2, "length_m": 20.0, "completed": True},
         {"team": "A", "f0": 250, "from": 2, "to": 1, "length_m": 35.0, "completed": True}]
    ours = [_p(0.9, 1, 2, fm=(0, 0), tm=(18, 0)), _p(9.8, 2, 1, fm=(0, 0), tm=(31, 0)), _p(1.0, 5, 6, team="B")]
    r = NP.match_lengths(T, ours, f0=0)
    assert r["n"] == 2 and r["err_median_m"] == -3.0 and r["within_5m_pct"] == 100.0
    assert (r["long_truth"], r["long_ours"], r["long_both"]) == (1, 1, 1) and r["same_pair_pct"] == 100.0


def test_truth_passes_reads_metrica_rows(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
                 "Home,PASS,,1,10,0.4,20,0.8,Player3,Player7,0.5,0.5,0.6,0.5\n"
                 "Away,PASS,,1,30,1.2,40,1.6,Player20,Player22,0.4,0.2,0.4,0.6\n"
                 "Home,BALL LOST,,1,50,2,60,2.4,Player3,,0.5,0.5,0.7,0.5\n"
                 "Home,PASS,,2,10,0.4,20,0.8,Player3,Player7,0.5,0.5,0.6,0.5\n")
    T = NP.truth_passes(str(p), 1)
    assert [(t["team"], t["from"], t["to"]) for t in T] == [("A", 103, 107), ("B", 220, 222)]
    assert abs(T[0]["length_m"] - 10.5) < 1e-9 and abs(T[1]["length_m"] - 0.4 * 68) < 1e-9
