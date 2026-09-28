# Tracking comparison, SFK-BP clip, 20 s of live play (28 Sep, free runner)
Note: in the "before" data the team labels are swapped (red circles sit on white players), so its dark/light rows are read the other way round below.

| | Before (app now) | New (RF-DETR + per-match kits + gap filling) |
|---|---|---|
| Dark players seen per frame | 5 | 8 |
| Light players seen per frame | 9 | 8 |
| Dark player stays tracked (median) | 0.5 s | 6.0 s |
| Light player stays tracked (median) | 2.2 s | 9.3 s |
| Track pieces in 20 s (fewer = less breaking up) | 91 | 42 |
| Referee counted as player (4 frames) | yes, in 2 | never |
| Goalkeeper counted (4 frames) | yes | no (bug: removed as "neither team") |
| Double circles on one player (4 frames) | 1 | 2 |

Graded by eye on 4 frames: new finds every visible outfield player in all 4 (before missed 2-4 per frame, mostly dark shirts, incl. a big close player).
