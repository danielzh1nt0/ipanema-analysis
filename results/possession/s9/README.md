# S9: a duel is not a lost ball (8 Oct, worker, local + free runner, $0)

**Rule (now the default, `P.confirm_losses`, run.py after passes):** a lost ball counts only when the team that won it
keeps it for at least 3 s (the loser doesn't get it back) or plays a pass before the loser gets it back. When a lost
ball is dropped as a duel, the one that hands the ball straight back is dropped too, so a duel counts zero, not one
loss for each team. `IPANEMA_LOSS_KEEP_S=0` = the old count.

## What it does to the numbers
On the exported demo matches (only the minutes covered by the frame files in the repo):

| match | minutes | lost balls before (A / B) | after (A / B) | change |
|---|---|---|---|---|
| SFK-BP 1st half | 42 | 74 / 72 | 55 / 54 | -25% |
| AIK 1st half (part) | 15 | 29 / 31 | 18 / 20 | -37% |
| Vallentuna | 81 | 188 / 188 | 146 / 151 | -21% |

That is the ~20-30% the 4 Oct check said the count was too high.

## Is it right? Blind check by eye (SFK-BP)
35 lost balls picked at random (20 the rule drops, 15 it keeps), shuffled, strips cut on the free runner
(`results/qa/s9/*.jpg`: 1 s before, the change, +1, +2, +3, +4.5 s), graded **before** opening the rule's verdicts
(`blind_grades.json`, key in `blind_key.json`):

| | real loss | duel | no change | can't tell |
|---|---|---|---|---|
| rule drops (20) | 2 | 9 | 4 | 5 |
| rule keeps (15) | 10 | 3 | 1 | 1 |

- Of the clear ones it drops, 13 of 15 were duels / no real change.
- Of the clear ones it keeps, 10 of 14 are real (was about 56% real across all lost balls, now about 71%).
- It loses about 1 real lost ball in 16. The 2 real ones it dropped: s9_22, a black win counted as 'the ball handed
  straight back' after an earlier duel, and s9_33, a clean white take where the rebuilt possession never gives white
  the ball (0 s held).

The older 36 spell strips (`results/kaggle/spell_zoom`, grades in `eye_grades.json`, short11-23 graded today):
of the 24 lost balls at the edges of spells graded as duels or noise, the rule drops 15.

## Pro data (Metrica, 4 halves, `metrica_*_*.json`)
- Clean positions: 242 -> 224 lost balls (labels: 490; 310 by the same 3-s / pass rule), share that match a
  labelled loss unchanged (75%), share of the labelled 3-s losses found 48% -> 45%.
- Noisy positions (streak): no change (95 -> 95).
- So it's close to neutral on pro data: the 1.5-s spell state already hides most pro duels. Our footage has the long
  scrappy contests, so that's where the rule matters.

## Not done / notes
- The app still shows the old numbers and the label "Losses (incl. duels)" until the next paid run (Daniel's go).
  After that the label can go back to "Losses".
- The check uses the exported possession (every 3rd frame) rebuilt into the counted state; it agrees with the
  exported turnovers 92-96% of the time (AIK 83%). The pipeline uses the exact state.
- Turnover payloads get `winner_kept_s` and `confirmed` (hold / pass / clip end).

Files: `ipanema/possession.py` (confirm_losses), `ipanema/run.py`, `tools/metricalab.py` (loss_keep_s),
`tools/s9lab.py`, `tools/s9_strips.py`, `tests/test_s9_confirm_losses.py`, `tests/test_fullmatch.py` (the rule runs in the
end-to-end dry run).
