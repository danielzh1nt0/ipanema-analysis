# N0-29a (29 Oct, worker, local $0, nothing on Modal): per-player pressures and time under pressure on pro data

Card: no 29 Oct card yet (00:20 UTC, before 03:00); the 28 Oct card's 'worse' rows were already explained and fixed by
N0-28c/d. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before)
`stats.json players[]` (both hidden in the app until M2, because a 'player' is still a track piece):

- `pressures_applied` - per opponent possession sequence, the player counts once if he comes within 2 m of the carrier
  (S7). N0-24d checked only the team sum (0.76-1.07x).
- `time_under_pressure_s` - his seconds on the ball (inside his cleaned contacts, N0-26d) with an opponent within 2 m.

Question: **once M2 gives one id per player, are they right per player?** Metrica's 2 pro games, 4 halves, real ids,
players with 10+ minutes (~11-12 per team-half). Ours = the pipeline on Metrica's positions with a perfect ball and our
two ball-noise models (streak, flicker - flicker is the one like our clips; 2 seeds), `analytics.stats` exactly as
`run.py` calls it. `tools/n0_29a_playerpress.py` -> `metrica_playerpress.json`.

Truths:
- **same rule**: our definition with perfect inputs - Metrica's sequences, the carrier from the events, but only while
  the tracked ball is within 2.5 m of him (the app's carrier needs that too), 2 m from tracking.
- **defensive actions**: the usual per-player count - tackles + ground challenges + interceptions + steals.
- **baseline 'near the play'**: seconds within 10 m of the ball. A per-player figure that ranks players no better than
  this says only 'he was near the ball'.

## Result (mean over team-halves)

| | perfect ball | streak noise | flicker (like our clips) |
|---|---|---|---|
| **pressures**: ours / truth | 1.02x | 0.82x | 1.22x |
| per-player error (median) | 18% | 29% | 22% |
| order of the players (rank) | 0.86 | 0.77 | 0.80 |
| truth's 3 busiest in our top 3 | 1.9 / 3 | 1.8 / 3 | 2.1 / 3 |
| *baseline 'near the play' rank* | 0.59 | | |
| pressures vs tackles + interceptions (rank) | 0.52 | 0.38 | 0.49 |
| *same rule, perfect inputs vs tackles + interceptions* | 0.34 | | |
| *baseline 'near the play' vs tackles + interceptions* | 0.48 | | |
| **time under pressure**: ours / truth | 1.25x | **0.25x** | 0.99x |
| per-player error (median) | 20% | 78% | 17% |
| order of the players (rank) | 0.94 | 0.84 | 0.86 |
| share of his time on the ball under pressure, ours vs truth | 17 vs 19% (2 pts off) | 8 vs 19% (10 pts) | 16 vs 19% (3 pts) |

- **Pressures per player hold** (picture `players.png`, top row): about the right number, ~1 in 5 off per player, the
  right order, and they say more than 'who was near the ball' (0.77-0.86 vs 0.59).
- **But they are not a defending count.** Against tackles + interceptions they rank players no better than 'near the
  play' (0.38-0.52 vs 0.48), and even our rule with perfect inputs only 0.34. Name it 'pressures (within 2 m)', never
  'defensive actions' or 'duels'.
- **Time under pressure per player holds with a perfect or flickering ball** (17-20% off, rank 0.86-0.94, share 2-3 pts),
  but with streak noise (ball missing in runs, 35% of players missing) it reads a quarter of the truth: few seconds
  survive as cleaned contacts. The order still holds (0.84). Our clips look like flicker, so it should be close - after M2.
- Not tested: our real camera views (players out of the picture). N0-23b found ~1 in 6 presses missed that way, so
  expect pressures ~15% lower per player on our footage.

## Tried, not switched on
- Time under pressure from every carrier frame (N0-26d's old on-ball rule) instead of the cleaned contacts: better with
  streak noise (16 of 16 team-halves), worse with a perfect ball (0 of 8) and flicker (3 of 16) -> today's rule stays.
- A first truth that took the event carrier alone (he 'has' the ball from receiving it to his next event, also through
  stoppages and long balls) made time under pressure look 0.45-0.66x and both rules look bad; kept for the record in
  `metrica_playerpress_eventcarrier.json`, not used.

## What it means
No pipeline change, app and export unchanged. Both per-player figures are worth showing once M2 joins the track pieces
(on today's pieces the numbers are spread over hundreds of pieces: SFK-BP clip 62 of 273 pieces with a pressure, top
6). Nothing for Daniel to decide now.

## Files
- `tools/n0_29a_playerpress.py` (lab; `--dry` stand-ins; `--refresh-ball 2.5` re-scores the saved runs with the
  ball-at-his-feet truth), `tools/n0_29a_plot.py`.
- `metrica_playerpress.json` (summary + 20 runs with every player), `metrica_playerpress_eventcarrier.json`,
  `players.png` (ours vs truth per player, seed 0).
- `tests/test_n0_29a_playerpress.py` (6): defensive-action types, same-rule truth once per sequence / only within 2 m,
  ball-within-2.5 m truth, team scores, a missing player counts 0 and < 10 min is left out, dry run + summary.
