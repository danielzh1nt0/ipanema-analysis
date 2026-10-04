# P2b - Reymersholm whites vanish in tracking (4 Oct, worker, local + free GitHub runner, $0, no Modal)

**Cause.** The kit model learns team colours only from people whose feet stand on grass. At night the floodlit grass
is much lighter than the reference colour, so this grass test kept only 12-29% of the people at Reymersholm (108 of
895 in piece 2227). From so few people the white team did not form its own colour group; whites read "neither team"
and the clean-up removed them as referee/staff. (The P2 note blamed the pitch-edge test - it was the grass test.)

**Fix** (ipanema/kits.py, `KitTeamModel.fit_frames`): when the grass test keeps less than 35% of the people, the kit
model learns from the people the pitch-edge test keeps instead (log line "grass test keeps only ... -> pitch-edge test
instead"). Env `IPANEMA_PITCH_FALLBACK` (share, default 0.35; 0 = old behaviour). Other grounds keep 43-88% -> no change.
Test: tests/test_p2b_pitch_fallback.py.

**Offline check** (tools/p2blab.py, kit model fitted on each P2 piece's 8 key frames; p2blab.json):
- 252 Reymersholm night players labelled by eye, team right: piece 726 74 -> 220, piece 2227 173 -> 188, piece 4227 125 -> 150.
- The other 15 pieces (Solheim, Djursholm, Spånga, Vasalund, Solberga): identical numbers (the fallback never fires).
- Sheets: `*_now_teamB.jpg` mixes greens and whites; `*_fix_teamA/B.jpg` are clean green / white.

**Free-runner re-track** (tools/p2b_track.py, same 3 x 20 s as P2; track/<piece>/), players per frame dark / light:

| piece | before (P2) | after | by eye (compare_*.jpg) |
|---|---|---|---|
| 726 | 12 / 1 (no split: everyone "dark") | 5 / 6 | whites and greens now split right, ~1 wrong per frame |
| 2227 | 9 / 2 | 8 / 9 | almost all right; referee sometimes counted light |
| 4227 | 5 / 1 | 6 / 2 | still bad: several greens read as the white team, spectators by the right-hand fence tracked as players. Same as before -> P2c |

Not in the app until the next app run (Modal, waits for Daniel).
