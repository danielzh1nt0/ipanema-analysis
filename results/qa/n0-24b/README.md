# N0-24b (24 Oct, worker, local $0): high turnovers and field tilt on pro data

Card: 24 Oct card says nothing worse than 23 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

Never checked against a truth before:

- `metrics.high_turnover_counts` / `metrics.high_turnovers` - balls won, zone from the ball's spot in the winner's
  attacking direction; 'high' = won in his final third (the app's High turnovers).
- `metrics.field[team].field_tilt_pct` - share of the team's ball-control time with the ball in its final third.

On Metrica's 2 pro games, 4 halves (`tools/n0_24b_hightov.py` -> `metrica_hightov.json`; summary
`tools/n0_24b_summary.py`). Truth = Metrica's events + tracking: a ball won = a team's chain of on-ball events starting
straight after the other team's chain in open play; 'kept' = what our rules can count (loser had it >= 1 s, winner keeps
it >= 3 s or passes). Ours = the pipeline on Metrica's positions, perfect ball and our two ball-noise models (2 seeds).

## Result

**Field tilt holds.** Ours minus truth, points, over the team-halves: perfect ball -4 / -0.6 / +3 (min / mean / max),
streak noise -5 / +0.7 / +7, flicker -5 / -2.1 / +5. Attack direction right in every half.

**High turnovers do not.** Totals over 4 halves (noise rows = 2 seeds, so twice the truth):

| ball | ours high | of them real | truth high (kept) | truth high (all) |
|---|---|---|---|---|
| perfect | 14 | 8 | 43 | 82 |
| streak noise | 22 | 1 | 86 | 164 |
| flicker noise (like our clips) | 59 | 10 | 86 | 164 |

- With a perfect ball we count about **1 in 3** of the real high turnovers, and ~6 in 10 of ours are real.
- With our clips' kind of ball noise the ones we show are **mostly not real** (1 in 6 or worse).
- The cause is upstream, not the zone: we find only 140 of 326 kept balls won at all (perfect ball); when we do find
  one, its zone is right 129 / 140. High wins are missed more than others - they are mostly interceptions of a pass, and
  the loser 'had' the ball under 1 s by our possession state (pass in flight = loose).

## Tried, not switched on

Lab: loosen the turnover rule's 'loser had it >= 1 s' (`--before`, `metrica_hightov_lab_before.json`):

| loser hold | perfect ours / real | streak ours / real | flicker ours / real |
|---|---|---|---|
| 1 s (today) | 14 / 8 | 22 / 1 | 59 / 10 |
| 0.5 s | 19 / 10 | 26 / 2 | 67 / 11 |
| 0.2 s | 19 / 10 | 31 / 3 | 75 / 15 |

A small gain with a clean ball, mostly extra fakes with noise; it would also move balls lost, counter-press and every
stat built on turnovers. **No pipeline change.** The real fix is the same as P-PASS / N0-22f / N0-24a: a steadier ball.

## Built, off by default

`IPANEMA_HIGH_TOV=withhold` writes `high_turnover_counts` as null (app shows a dash); the list of balls won (with
zones) and field tilt are untouched. Question under Waiting for Daniel. (The UI contract has had high turnovers as
HIDE since 2 Oct.)

App copies today (high A / B, tilt A / B %): SFK-BP half 13 / 16, 24 / 35; SFK-BP clip 0 / 3, 12 / 44; AIK half 27 / 10,
35 / 16; Vallentuna 27 / 30, 25 / 18.

Tests: `tests/test_n0_24b_hightov.py` (5): zone follows the winner's direction, truth balls won (open play, kept flag,
none after a throw-in), truth tilt skips dead time, the matching counts, the withhold switch. Suite 585 pass + 7
skipped; 3 fail here only because this machine has no `supervision` package (tracking tests, untouched).
