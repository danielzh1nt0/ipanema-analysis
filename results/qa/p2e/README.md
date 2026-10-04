# P2e - Reymersholm 4227: the pink referee and whites read as the green team (4 Oct, worker, local + free runner, $0, no Modal)

**Cause (referee + near whites).** Not the colour reading: the referee's own track reads 'neither team' in 96 of 110
checks, the near white at the left edge reads white 35/37. The **keeper rules** in `tracking.clean` made them keepers:
a ground without calibration has screen positions, not metres, so "in the left goalmouth" just meant "at the left edge
of the picture", and every such track was forced into the team defending left (green). On 4227: 5 'keepers'
(referee, near white, far white...). Across all 18 P2 pieces the rule made 48 edge 'keepers', 30 of them against their
own colour vote.

**Fix.** `tracking.clean(..., keepers_by_zone=False)` skips both goalmouth keeper rules; tracktest passes it when the
ground has no calibration (`IPANEMA_SCREEN_KEEPERS=1` = old). The app always has calibration, so it is untouched
(default `True`). Test: tests/test_p2e_screen_keepers.py. Suite: 243 passed, 1 failure that also fails without this
change (test_panorama_extend, local OpenCV 5).

**By eye on the other pieces** (`tools/p2elab.py` -> `p2elab.json`, sheet `edge_keepers.jpg`, team colours from each
piece's kit strips): of the 30 edge 'keepers' that change, **24 right** (14 players back in their own kit's team,
10 bench people / spectators / the referee now dropped as 'neither'), **2 worse** (an orange-kit keeper on Djursholm
5072 and two Vasalund players on 1 weak reading now dropped), 4 can't tell (crop between two players).
Cost: on uncalibrated grounds real keepers in their own kit are no longer kept (the rule could not find them anyway).

**Free-runner re-track of 4227** (tools/p2e_track.py, same 20 s as P2-P2d, 10 min, $0; graded with
`tools/p2e_grade.py` on the 88 labelled people):

| key frames, 88 labelled | P2d | now |
|---|---|---|
| referee / keepers read as green | 5/12 | **2/12** |
| whites read as green | 17/40 | **15/40** |
| whites right | 23/40 | 25/40 |
| greens right | 23/24 | 23/24 |
| spectators read as green | 3/11 | 3/11 |
| per frame dark / light (median) | 6 / 3 | 5 / 3 |

By eye (`track/.../compare_*.jpg`): frame 341 the pink referee has no ring any more (was green), frame 0 the white at
the left edge is white (was green). The referee is still green at frames 170 and 256 (the first seconds of his track,
before 'neither' wins the vote).

**Not fixed:** the dim far whites (tracks 2, 3, 9, 16: their own colour reads green, cream shirt with green spill;
P2c tried many colour readings, none better) -> P2f, after the demo.
