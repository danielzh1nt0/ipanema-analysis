# N0-28a — reaction after a loss: pressed / jogged / stood (28 Oct, worker, local + free runner $0, nothing on Modal)

**Card:** no 28 Oct card yet (00:20 UTC, before the 03:00 nightly); 27 Oct card (rescued, N0-27c): nothing worse than 26 Oct.
M2 (main session) and PC2 (waits for M2) skipped.

**What it is:** every `turnover_lost` event carries `payload.reactions {pressed[], jogged[], stood[]}` - the players of the
team that lost the ball, over the first 2 s after the win - and `stats.players[].counterpress_rate` = pressed / losses_seen.
Never checked before.

**Old rule** (`IPANEMA_REACTION=gap`): pressed if he ends within 5 m of the ball or the GAP to the ball shrank >= 2 m/s,
jogged >= 0.7 m/s, else stood. The gap also shrinks when the BALL comes to him (the winner runs or passes at him), and
grows when he sprints at a ball that is carried away.

## 1. The rule on perfect inputs (Metrica, 2 games, 385 turnovers, 3,850 player-reactions, keepers out) - `tools/n0_28a_reactions.py --part 1`

Truth for "did he react" = his own run toward the ball, measured frame by frame (`own_label`).

| | pressed | jogged | stood | agrees with his real run |
|---|---|---|---|---|
| old (gap) | 38.7% | 11.0% | 50.3% | **43%** |
| real run (frame by frame) | 31.8% | 32.4% | 35.8% | - |
| **new (`own`, from the 2 end frames only)** | 29.4% | 31.0% | 39.6% | **92%** |

- 26% of the old rule's 'pressed' were not running at the ball at all (some running AWAY at 3 m/s while a long ball came their way).
- 35% of the players who did run at the ball >= 2 m/s read 'stood' (the ball was carried away faster).
- Same on all 4 halves (old 42-45% agreement).

New rule (`possession.reaction_label`, mode `own`): his move (p2 - p0) / 2 s along the direction to the ball, averaged over
the start (b0 - p0) and the end (b2 - p2). Needs the same two frames as before, so the same players get a label.

**Side finding (not a fault of the new rule):** with perfect inputs the old 'pressed' count goes with winning the ball back
within 5 s (2+ pressers: 43% regained vs 0-1: 23%) and the real-run count does not (39% vs 37%). The old count partly
measures "the ball came back toward us", which is what a quick regain is. Real pressing effort in the first 2 s does not
by itself predict a regain in these two pro games. Player rates are about as repeatable half to half (Spearman old 0.46,
real run 0.40, new 0.36, 39 players).

## 2. Through our inputs (pipeline's own turnovers, matched to the truth ones) - `--part 2` (56 runs)

| our inputs | players of the losing team labelled | old: agrees with the real run | **new: agrees with the real run** | pressers per loss, abs. error old / new |
|---|---|---|---|---|
| perfect ball, all players | 100% | 43% | **80%** | 1.4 / 0.7 |
| flicker ball (like our clips) | 100% | 41% | **72%** | 1.9 / 1.0 |
| our camera views | 39% | 47% | **74%** | 2.7 / 2.0 |
| views + flicker | 37% | 45% | **73%** | 2.7 / 1.8 |

Better on every run. Through the camera only ~2 in 5 of the losing team get a label at all (the rest are out of the picture).

## 3. App clips (exact inputs, `tools/n0_28a_clips.py` -> `clips.json`)

| clip (5 min) | losses | player-reactions | old P / J / S | new P / J / S | label changes |
|---|---|---|---|---|---|
| SFK-BP | 17 | 101 | 18 / 9 / 74 | 16 / 28 / 57 | 49 |
| AIK | 17 | 85 | 19 / 11 / 55 | 11 / 8 / 66 | 38 |

## 4. By eye - blind sheets (free runner, `tools/n0_28a_frames.py`, results/free/n0-28a)

32 sheets (24 whose label changes + 8 unchanged, seeded), 6 tiles over the 2 s, graded before opening `key_SEALED.json`
(`grades.json`):

- Changed and clear by eye (13): **new right 9, old right 4, both wrong 0.**
- 5 of the 24 changed ones are not players: spectators behind a goal line / a person crouching at the far wall (3 of them)
  and one ID slide. The old rule called **all 5 'pressed'** (the ball came their way); the new one 'stood'.
- 6 changed ones: can't tell (camera pans as much as the player moves).
- Unchanged (8): 7 stood and both rules said stood, 1 can't tell.

## Result

New default `IPANEMA_REACTION=own` (`gap` = old). Only `payload.reactions` and `players[].counterpress_rate` change; turnovers,
time to press, near_at_2s, regains are untouched. Reaches the app at the next run. Still: only ~40% of the losing team is in
the picture, players are track pieces until M2, and spectators near a line still get a label (TOUCHLINE).
tests/test_n0_28a_reactions.py (7); full suite 709 pass, 4 skipped.
