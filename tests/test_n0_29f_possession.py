"""N0-29f: whole-half possession % lab helpers + the app's possession_pct rule (raw state, loose frames left out)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import n0_29f_possession as Z


def test_share_leaves_loose_out():
    assert Z.share([0, 0, 2, 1, 3]) == round(100 * 2 / 3, 1)
    assert Z.share([2, 2, 3]) is None


def test_fill_loose_credits_last_team():
    assert list(Z.fill_loose([2, 0, 2, 2, 1, 2])) == [2, 0, 0, 0, 1, 1]


def test_gap_mask_marks_only_long_gaps():
    hb = [1, 0, 1, 0, 0, 0, 1]
    assert list(Z.gap_mask(hb, 1.0, 2)) == [0, 0, 0, 1, 1, 1, 0]
    assert not Z.gap_mask(hb, 1.0, 4).any()


def test_variants_dry():
    st = np.array([2, 0, 0, 2, 2, 1, 1, 1, 2, 0])
    v = Z.variants(st, st, [(3, 4)], [1, 0, 0, 0, 1, 1, 0, 1, 1, 1], fps=1.0)
    assert v["raw"] == 50.0 and v["raw_nostop"] == 50.0 and v["raw_fill"] == 42.9 and v["raw_gap2"] == 25.0


def test_truth_share_same_frames():
    own = {10: "A", 11: "A", 12: "B", 13: "B"}
    assert Z.truth_share(own, 10, 4) == 50.0
    assert Z.truth_share(own, 10, 4, keep=np.array([True, True, True, False])) == round(100 * 2 / 3, 1)


def test_app_possession_pct_is_raw_share():
    """analytics.stats writes A's share of the frames the raw state gives to A or B - what the lab calls 'raw'"""
    from ipanema.possession import STATES
    st = [0] * 30 + [2] * 20 + [1] * 10
    poss = [STATES[s] if s < 2 else None for s in st]
    ctrl = sum(1 for p in poss if p is not None)
    assert round(100 * sum(1 for p in poss if p == "A") / ctrl) == round(Z.share(st))


def test_card_row_keeps_parsing_with_seen_note():
    """the nightly card's pro-data row gains '(x where Metrica has a ball)'; read_card must still find both numbers"""
    import nightly_scorecard as NS
    q = {"stats_pro_data": {"worst_pass_err_pct": 15, "worst_poss_err_pts": 8, "worst_poss_err_seen_pts": 2.4}}
    lines = NS.build_lines("2026-10-30", q, {})
    row = next(l for l in lines if l.startswith("| Stats on pro data"))
    assert "2.4 where Metrica has a ball" in row
    got = NS.read_card([row]); assert got["pass_err"] == 15 and got["poss_err"] == 8
    q["stats_pro_data"].pop("worst_poss_err_seen_pts")            # old quick.json: row as before
    row = next(l for l in NS.build_lines("2026-10-30", q, {}) if l.startswith("| Stats on pro data"))
    assert row.endswith("possession 8 pts off |")
