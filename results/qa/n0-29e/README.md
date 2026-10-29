# N0-29e (29 Oct, worker, local $0, nothing on Modal): balls lost / won per team and passes per sequence

Card: 29 Oct card says nothing worse than 28 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before as the app computes them today)
- `stats.json teams[].losses` / `recoveries`: the kept turnovers (S9 rule: the winner keeps it 3 s or plays a pass)
  lost / won by each team. These are also the app's 'lost the ball' events (turnover_lost, beta) and the match-flow dots.
  `losses A = recoveries B` by build, so one count per team is graded.
- `teams[].passes_per_sequence` (all passes tried / sequences; the UI contract hid it on 2 Oct because sequences
  flickered - S8 has fixed that since).

S9 (8 Oct) only compared totals on one setup. Nobody had matched our losses to the real ones one by one.

## How (no clicking)
Metrica's 2 pro games, 4 halves, hand-labelled events. Truth = every BALL LOST of a team in open play (END HALF /
OFFSIDE / REFEREE HIT left out), and the S9 kind of it (the winner's next chain lasts 3 s or has a pass). Ours =
`analytics.stats` on Metrica's positions as the pipeline sees them (`n0_24d_ppda.ours_run`), perfect ball, our
'flicker' ball (dot hops to a nearby opponent, like our clips, 2 seeds) and 'streak' noise (2 seeds). A loss of ours
and a real one of the same team within 2 s = found. `tools/n0_29e_lossrec.py` -> `metrica_lossrec.json`,
picture `losses.png`.

| ball | our count vs every real loss | vs S9-kind losses | real S9-kind losses found | ours that are real | team losing more right |
|---|---|---|---|---|---|
| perfect | **0.47x** (0.32-0.56) | 0.74x | 44% | 75% | 2 of 3 |
| flicker (our clips) | 0.56x | **0.87x** | 33% | **47%** | 3 of 6 |
| streak | 0.17x | 0.27x | 12% | 53% | 4 of 6 |

Plain words: even with a perfect ball we count about half of the real balls lost, and 3 in 4 of the ones we show are
real. With our kind of ball the total lands near the S9 kind by luck - half of what we show is not a real loss and
two in three real ones are missed. Which team lost the ball more is a coin flip.

Passes per sequence: ours 4.7 vs truth 3.2 (1.5x, any ball model but streak) - our sequences join across loose
balls (spell state) so there are fewer of them than real chains. **Keep it hidden.**

## Why so many are missed (lab, `tools/n0_29e_lab.py` -> `lab.json`)
The turnover step was re-run on the captured inputs with 60 settings (spell state take 1.5/1.0/0.5/off, join 3/2,
before/after hold 1/0.5 s, S9 keep 3/2/off). Nothing fixes it:
- perfect ball: the best setting finds 59% of the S9-kind (today 45%) but only 67% of ours are real; even the raw
  state with no S9 rule finds 57%.
- flicker: every looser setting raises the count fast (up to 5.7x) while the real share drops to 15-41%.
Looking at the missed ones (Game 1, first half: 44 of 79 missed, 35 found, all found within 1 s): the raw possession
state itself says 'loose' around them - long balls after throw-ins, goal kicks and free kicks, headers and
contested balls where nobody is within 1.5 m of the ball for seconds. The ball was visible (87% of frames).
A wider 'on the ball' radius (`tools/n0_29e_nearm.py`, IPANEMA_NEAR_M) trades the same way: 3.0 m finds 51% (perfect)
but 65% real, flicker 45% real and the count 1.24x - no clear win, so nothing changed.

## Result
Measurement only, no code change in the pipeline. Our footage is harder (ball in the air, far players), so the app's
'balls lost' count and the lost-ball dots are rough: about half the real ones, and on our clips about half of those
shown are not real. Decision for Daniel (under 'Waiting for Daniel'): keep them as Beta with a 'rough' note, or hide
the count.

Repo copies for context (old exports): SFK-BP first half 60 / 58 lost, AIK 63 / 70, Vallentuna 133 / 137.

Tests: `tests/test_n0_29e_lossrec.py` (4): truth losses + S9 kind + pass tries on stand-in events, matching (team,
2 s window, one-to-one), dry run of the scores, and the app's team row build (losses A = recoveries B, no sequences ->
passes per sequence null).
