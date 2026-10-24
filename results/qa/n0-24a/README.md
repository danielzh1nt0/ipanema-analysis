# N0-24a (24 Oct, worker, local $0): pass completion % on pro data

Card: no 24 Oct card yet (00:20 UTC, due after 03:00); 23 Oct card says nothing worse than 22 Oct. M2 (main session)
and PC2 (waits for M2) skipped.

## What was checked

`stats.json teams[].pass_completion_pct` (share of a team's passes whose receiver is a team-mate, `analytics.passes`)
had never been checked against a truth. Checked on Metrica's 2 pro games, 4 halves (`tools/n0_24a_completion.py` ->
`metrica_completion.json`). Truth = Metrica's events: completed = PASS, failed = BALL LOST that is a pass (not a
dribble lost: THEFT / FORCED, not end of half / offside / referee). 'all' also counts a BALL OUT as a failed pass.
Ours = the pipeline's passes on Metrica's positions, perfect ball and our two ball-noise models (2 seeds each).

## Result

Team completion %, ours minus truth, points (8 team-halves; min / mean / max):

| ball | vs truth (pass to opponent) | vs truth incl. ball out |
|---|---|---|
| perfect | -3 / -1 / +4 | 0 / +3 / +7 |
| 'streak' noise | -4 / 0 / +4 | -1 / +3 / +10 |
| 'flicker' noise (like our clips: the ball dot hops to a nearby opponent) | **-23 / -16 / -7** | -20 / -13 / -1 |

- With a clean or streaky ball the team figure holds (within ~4 points of the truth).
- With the ball flicker our clips have, it reads **~16 points too low**: every hop to an opponent and back is two
  'failed passes' (ours: 2,144 failed vs truth 834 over the 4 halves x 2 seeds).
- Per pass it is weak even with a perfect ball: only 96 of 417 truth failed passes are matched to a pass we call
  failed, and about 1 in 5 of our failed passes is matched to a truth failed pass. On one half (Game 1, 1st, perfect
  ball) the nearest truth event within 1 s of our 119 failed passes: a pass 38, a ball lost / interception 47, nothing
  19, a throw-in 6, other 9 - so at most ~2 in 5 are real. So the team % is right with a clean ball partly because errors cancel.
  The per-pass 'bad_lost' / 'execution_error' labels should not be trusted.

## Tried, not switched on

Lab rule (`drop_handbacks`, tools only): a pass to an opponent who plays it straight back within K s is a
flicker/duel, not two failed passes -> both dropped. Mean error, points:

| K | perfect | streak | flicker |
|---|---|---|---|
| off (today) | -1.0 | -0.4 | -16.4 |
| 1 s | +2.2 | +2.2 | -10.1 |
| 2 s | +4.2 | +4.1 | -5.5 |
| 3 s | +4.5 | +4.6 | -4.3 |

It moves the flicker case towards the truth but pushes the clean cases ~4 points high, and our failed passes are
still only ~13% real under flicker. Not better on all rows -> **no pipeline change**. The real fix is a steadier ball
(same as P-PASS / N0-22f / N0-23a).

## Passes per sequence

Not comparable to a truth: Metrica's event chains (truth 1.8-3.2 passes per chain) and our sequences are different
things; ours reads 1.7-2.2x with a perfect ball, 0.9-1.6x with streak noise, 1.9-2.8x with flicker. Left as is.
Also seen: the repo copies' `teams[].sequences` (e.g. SFK-BP clip 112 / 102) do not match their own `sequences`
list (40 / 41) - stale mixed copies (N0-19c), not checked further.

App copies today: completion SFK-BP clip 73 / 68%, AIK clip 78 / 82%, SFK-BP half 86 / 81%, Vallentuna 81 / 86% -
by the flicker result these probably read low. Completion % stays Beta; question under Waiting for Daniel.

Tests: `tests/test_n0_24a_completion.py` (4): a pass to an opponent is not completed and the team %, the Metrica
truth attempt rules (theft / woodwork left out, ball out only in 'all'), attempt matching, the lab hand-back rule.
