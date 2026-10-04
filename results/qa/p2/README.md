# P2 - players on the 6 training matches (4 Oct, free GitHub runner, $0, no Modal)

20 s tracked at 3 live-play spots per match (RF-DETR on CPU, same kit step / clean-up / gap filling as the app's
tracktest). Spots picked by eye from results/qa/players frames. Tool: tools/p2_track.py (3 parallel free-runner jobs,
~2 h), table: tools/p2_table.py. "Dropped" = people the detector finds on the 8 key frames of a piece that no tracked
player stands on (dropped_sheet.jpg in each piece, numbered crops).

| ground | players seen per frame (dark / light, 3 pieces) | median s a player stays tracked | dropped on 8 frames: real players (by eye) |
|---|---|---|---|
| Reymersholm (night) | 9/2, 5/1, 12/1 | 5-13 | 108 in piece 2227: ~60 white players, ~8 referee, ~40 bench/spectators |
| Vasalund | 6/7, 6/5, 7/8 | 5-8 | 77 in piece 1164: ~20 players by the far touchline, rest bench/subs/referee |
| Solberga (floodlights) | 7/7, 8/8, 7/8 | 3-8 | 123 in piece 1500: ~12-15 players, ~100 spectators in the stand, referee |
| Djursholm | 9/8, 9/10, 6/6 | 3-9 | not graded |
| Spånga (night) | 9/7, 8/7, 8/5 | 1-6 | not graded |
| Solheim | 7/5, 8/8, 7/8 | 3-8 | not graded |

Full per-piece numbers: table.md / table.json.

**Worst ground: Reymersholm - the white team almost disappears (1-2 per frame, should be ~8).** It is not whites read
as green: the whites are thrown away. Why (piece logs): the pitch-edge test (P7) leaves out most people as off the pitch
(787 of 895, 655 of 848, 283 of 397), so the kit model learns from only 108-193 people, reads whites as "neither team", and clean-up then
removes them as referee/staff (5,460 rows in piece 2227). P8's 239/252 was on single key frames, not in tracking.
Fix -> P2b (not done in this run: the 3-hour limit).

Referee: on the 3 graded grounds the referee shows up among the dropped people (orange Reymersholm, blue Vasalund
and Solberga) = rightly not counted as a player in those frames.
Speed: ~15 min per 20-s piece on the free CPU runner.
