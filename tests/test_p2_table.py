"""P2 (4 Oct): the 'dropped by tracking' count (detector people with no tracked player at their feet) and the piece list
of the free-runner job (6 matches x 3 spots, 3 groups of 2 matches)."""
import os, sys, json, gzip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "tools"))
import p2_table as T, p2_track as P

def test_dropped_matches_by_feet():
    dets = [[100, 100, 120, 160], [300, 300, 330, 390], [500, 50, 510, 80]]
    rows = [[1, "A", [110, 158], False],          # on det 0's feet
            [2, "B", [318, 395], False],          # within 0.6 x 90 px of det 1's feet
            [3, "A", [505, 80], True]]            # filled row on det 2: does not count as tracked
    assert T.dropped(dets, rows) == [2]
    assert T.dropped(dets, []) == [0, 1, 2]
    assert T.dropped(dets, [[4, "A", None, False]]) == [0, 1, 2]

def test_far_player_not_matched():
    assert T.dropped([[0, 0, 10, 30]], [[1, "A", [40, 30], False]]) == [0]

def test_groups_cover_six_matches_three_spots():
    allp = [p for g in "abc" for p in P.pieces(g)]
    assert len(allp) == 18 and len({m for m, _ in allp}) == 6 and len(set(allp)) == 18
    assert all(len(P.pieces(g)) == 6 for g in "abc")

def test_piece_stats_and_table(tmp_path):
    d = tmp_path / "m_100"; d.mkdir()
    json.dump({"new, RF-DETR": {"observed_per_frame": {"dark": 7.0, "light": 8.0}, "shown_per_frame": {"dark": 8.0, "light": 8.0},
               "tracks": 30, "median_track_s": {"dark": 3.0, "light": 4.0}, "minutes": 1.0, "filled_rows": 2}}, open(d / "summary.json", "w"))
    json.dump({"0": [[100, 100, 120, 160], [500, 50, 510, 80]]}, open(d / "keydets.json", "w"))
    json.dump({"new, RF-DETR": {"0": [[1, "A", [110, 158], False]]}}, open(d / "rows_keyframes.json", "w"))
    with gzip.open(d / "rows_all.json.gz", "wt") as fh: json.dump({"fps": 2.0, "rows": {"new, RF-DETR": {"0": [[1, "A", [1, 1], False], [2, "B", [2, 2], True]]}}}, fh)
    res = T.main(str(tmp_path))
    s = res["m_100"]; assert s["det_people"] == 2 and s["dropped"] == 1 and s["player_s"] == 0.5
    assert "m_100" in open(tmp_path / "table.md").read()
