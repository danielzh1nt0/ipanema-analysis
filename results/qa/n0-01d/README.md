# N0-01d (1 Nov 2026): places in the app copies

Worker, free runner + local, $0, nothing on Modal. 1 Nov card: nothing worse than 31 Oct (tests 204/204). M2 / PC2 skipped.

## What was checked
`tools/n0_01d_places.py` checks every spot, length and place-label in the 4 app copies (results/volume/runs/matches/*/stats.json)
against the pitch and against each other (~20 rules: spots on the pitch within 4 m, zone vs x, restart kind vs spot, shot
distance vs spot, pass length / gain / kind / progressive, run distance / speed, sequence duration, entries counts, tilt,
heat maps sum to 1, pass network vs passes). Result: `places.json`. Lengths, gains, kinds, zones, durations, counts and heat
maps all hold. What breaks is only **spots more than 4 m off the pitch**: SFK-BP 7, clip 3, AIK 5, Vallentuna 12
(final-third entries in there are already fixed by N0-31e and reach the app at the next run).

## Blind check by eye (16 sheets, free runner)
8 off-pitch spots + 8 on-pitch controls of the same kinds, shuffled, 6 frames each with the exported spot drawn as a cross
through each frame's own calibration (`results/free/n0_01d/m*.jpg`). Graded before reading the flags
(`grades_blind.json`; m00-m03 flags were glimpsed while reading the job files).

| | flagged off pitch (8) | controls on pitch (8) |
|---|---|---|
| real ball on the pitch by the line, spot pushed out | 3 (ball won 9.7 m, throw-in 5.7 m - a contest at the line, no throw seen, goal kick 7.5 m - real, ball at the 6-m box) | - |
| ball in the air, no restart at all | 2 ('throw-ins' 18.3 m and 9.4 m: long/high ball, camera up in the trees) | 0 |
| ball at the cross | - | 5 (throw-in, ball won, plausible throw-in, shot, goal kick) |
| ball elsewhere | - | 1 (ball won, ball 10-15 m along the line) |
| no throw taken although a 'throw-in' | (1, above) | 1 |
| can't tell / no calibrated frame | 3 | 2 |

So an off-pitch spot is never a real off-pitch place: when the event is real, the ball was on the pitch by the line (far-side
metres stretch, as in C3c/N0-31e); otherwise it was a ball in the air. Fake 'throw-ins' also exist among on-pitch spots
(1 of 8 controls), so the spot can't be used to drop restarts (too few to set a limit: real 7.5 / 9.7 m vs fake 9.4 m).

## Change
`ipanema/export.py clamp_all`: restarts, balls won, pressure points and shots are moved to the nearest point of the pitch
before export (raw kept as `x_raw_m` / `y_raw_m`, `spot_clamped: true`); a moved shot's distance is re-measured.
`IPANEMA_SPOT_CLAMP=0` = old. Event kinds, counts and times unchanged.

On the 4 copies (`tools/n0_01d_clamp.py`, `clamp.json`): moved restarts 14/66, 7/13, 2/42, 15/118; balls won 8/118, 1/20,
1/133, 17/270; shots 1/8, 0, 0, 2/16; pressure points 0. Places check broken 7 -> 1, 3 -> 1, 5 -> 5, 12 -> 10 (all left =
final-third entries, already fixed by N0-31e). Reaches the app at the next run.

tests/test_n0_01d_clamp.py (4), tests/test_n0_01d_places.py (4).
