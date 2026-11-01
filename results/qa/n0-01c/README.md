# N0-01c (1 Nov, worker, free runner + local, $0, nothing on Modal): passes from a player to himself

1 Nov card: nothing worse than 31 Oct (tests 204/204). M2 (main session) and PC2 (waits for M2) skipped.

N0-01b's side note: 4-8% of the passes in the app copies go from a player to himself (`from == to`: the ball moved 5 m or
more between two touches of one track). Checked for the first time.

## 1. Metrica pro data (real ids) - `tools/n0_01c_selfpass.py`, `metrica_selfpass.json`
Our passes matched to Metrica's pass attempts (same team, within 1 s), 4 halves, perfect ball + our two kinds of ball noise:

| ball | self passes (share of ours) | of them at a real pass | other passes real | drop them: found | real share | count vs truth | completion % error |
|---|---|---|---|---|---|---|---|
| perfect | 0.8% | 56% | 81% | 81.7 -> 81.3% | 81.1 -> 81.2% | 1.01 -> 1.00 | 8.9 -> 8.8 pts |
| 'streak' (ours) | 2.5% (both noises) | 28% (both) | 70% | 67.8 -> 67.1% | 70.5 -> 72.8% | 0.96 -> 0.93 | 2.0 -> 1.5 pts |
| 'flicker' | | | | 75.5 -> 74.8% | 68.5 -> 68.9% | 1.13 -> 1.12 | 6.9 -> 6.7 pts |

So a self pass is a real pass only 1 time in 2-4 (and then its receiver is wrong anyway); dropping them costs <1 pt of
found passes and makes ours a bit more often real and completion % a bit closer. With our ball noise the count drops 3 pts
(0.96 -> 0.93 of the truth): the nightly 'passes x% off' row may move 1-3 pts from this - expected, not a regression.

## 2. Our footage, blind by eye - `tools/n0_01c_moments.py`, `n0_01c_frames.py` (free runner), `grades_blind.json`
26 self passes (SFK-BP 8, Vallentuna 18) mixed with 12 normal completed passes; per pass 6 frames from the touch to the
next touch + 0.8 s, passer ringed 'P' only in the first frame; graded before opening `moments_key.json`.
Run 1 used one fixed box and lost the play when the Veo camera panned (sheets dropped); run 2 centres each frame on the
exported ball (`results/free/n0_01c`). Question: who has the ball next?

| | P himself (dribble / knock-on) | team-mate | opponent | P is a referee / bib | can't tell |
|---|---|---|---|---|---|
| 26 self passes | **8** | **0** | 1 | 1 | 16 |
| 12 normal passes | 3 | 4 | 1 | 1 | 3 |

Of the 10 decidable self passes none went to a team-mate. (Side note: 3 of 9 decidable normal passes were also the passer
running on with the ball - his track changed id between the two touches; M2's problem, not changed here.)

## Changes
- `ipanema/analytics.py` `IPANEMA_PASS_SELF` (default 0 = drop): a pass whose receiver is the passer's own track is no
  longer a pass. Reaches team totals, player cards, pass network and events at the next run (Modal is off, so it waits).
- `ipanema/metrics.py` `IPANEMA_NET_SELF` (default 0): the pass network skips a player -> himself link (a guard; the
  network drew loops on every copy).
- App copies (`tools/n0_01c_rebuild.py`, old rule reproduces the stored networks exactly first): self loops removed from
  the pass network - SFK-BP 6 + 2, clip 2 + 1, AIK 4 + 6, Vallentuna 13 + 17 links. Team totals in the copies are left as
  they are (changing them without a run would break the numbers-add-up sweep); the sweep still finds 0 broken rules on
  SFK-BP, AIK and Vallentuna (clip: only the known different-runs fault).
- `tests/test_n0_01c_selfpass.py` (7).

Expected at the next run, from the copies: passes Vallentuna 487 -> 450, SFK-BP 167 -> 159, AIK 225 -> 215; completion %
down slightly (66 of the 70 self passes on the copies count as completed: a team mostly "completes" a pass to the same player).
