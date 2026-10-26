# N0-25e: sequence labels - 'reached the final third' and 'how it ended' (26 Oct, worker, local + free runner, $0)

Card: no 26 Oct card yet (00:20 UTC, due after 03:00); the 25 Oct card says nothing worse than 24 Oct.

## What was checked
Every sequence in `stats.json -> sequences[]` carries `reached_final_third` and `end_reason`. Never checked before.

Two faults found:
1. **Reached the final third** was read only from the ball on the sequence's LAST frame. An attack that got into the
   final third and was pushed back, or lost the ball near the halfway line after it, said "no".
2. **End reason** never said "out of play / stoppage" on the counted (spell) state: the state that the sequences read
   hardly ever shows "dead", so a sequence ending with the ball going out was called "loose ball".

## Change (on by default; reaches the app at the next run)
- `reached_final_third`: the ball spent at least 0.5 s (summed) of the sequence in the team's final third, or was there on
  its last frame (the old rule is kept inside the new one). `IPANEMA_SEQ_REACH=end` = old, `IPANEMA_SEQ_REACH_MIN_S` = the 0.5 s.
- `end_reason`: a sequence that ends in a loose ball and runs into a stoppage (the restart step's own stoppages, >= 3 s,
  incl. the N0-24e unseen-ball rule) starting no more than 2 s after its last frame now ends "out of play / stoppage";
  one sequence per stoppage. `IPANEMA_SEQ_STOP_S=0` = old. New `P.stoppage_spans` returns the same stoppages `restarts()` counts.
- Wired in `ipanema/run.py` and, to match, `tools/metricalab.py` (its numbers don't change: it only counts sequence starts).

## Metrica pro data (`tools/n0_25e_sequences.py` -> `metrica_sequences.json`)
2 games, 4 halves, perfect ball + our 'streak' and 'flicker' ball noise (2 seeds each) = 20 runs. Truth = chains of one
team's on-ball events; "reached" = the clean ball in the team's final third on any frame; "stopped" = the chain ended
in a ball out, a foul or a set piece. Each true sequence matched to ours (same team, start within 2 s).

| rule | reach label right (matched) | real 'reached' found | ours 'reached' really reached |
|---|---|---|---|
| old (last frame) | 983 / 1346 = 73.0% | 406 / 700 | 406 / 475 |
| any frame | 1052 / 1346 = 78.2% | 495 / 700 | 495 / 584 |
| **>= 0.5 s (new default)** | **1037 / 1346 = 77.0%** | 473 / 700 | 473 / 555 |
| >= 1 s | 1031 / 1346 = 76.6% | 465 / 700 | 465 / 545 |

The 0.5 s rule is right at least as often as the old one on all 20 runs (better on 17). Share of sequences that reach it:
truth 43-50%; old 23-49%, new 27-53% (closer on the perfect and streak runs, a little high with flicker).

"Out of play / stoppage" (new): of our sequences given that end, 61% (perfect ball) / 58% (streak) / 53% (flicker) end
within -1..+4 s of a real stoppage, against 13-28% of the ones still called "loose ball" and 19-49% of interceptions - so the
label points at real stoppages about 2-4x more often than other ends, but roughly 4 in 10 are not one. It finds ~1 in 5
of the real stopped chains (67 / 312 matched): most stoppages are not seen by our ball.

## Our two app clips (`tools/n0_25e_clips.py` -> `clips.json`; free-runner sheets `results/free/n0-25e`, graded in `grades.json`)
The 25 Oct sheets were cut for the any-frame rule (17 sequences change; the rebuilt code gives exactly those 17).
By eye:
- reach, any frame (13): right 6, wrong 3, can't tell 4. The 3 wrong are 3-10 frames of false ball picks (a jump, the
  trees, the stand) - hence the 0.5 s minimum: it drops those 3 + 1 can't-tell and keeps all 6 right ones (3 can't tell kept;
  most "can't tell" sit on the far end of the 2/3 line at the far touchline, where the camera line is least sure).
- end "out of play / stoppage" (6): right 4 (two throw-ins, two goal kicks), 2 can't tell (the stoppage comes after the sheet).

Clip counts (old -> new): SFK-BP reached A 3 -> 5, B 6 -> 7; AIK A 3 -> 5, B 5 -> 9; ends 'out of play' SFK-BP 0 -> 5 (of 7
stoppages), AIK 0 -> 1. Nothing else in the export moves (spans, counts, events: only intercepted / dispossessed ends make
timeline events).

Tests: `tests/test_n0_25e_sequences.py` (7). Full suite 644 pass here, 3 fail only because this box has no `supervision`
(tiles / watchdog / v3 tests; same before the change).
