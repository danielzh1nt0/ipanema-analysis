# N0-29b (29 Oct, worker, local + free runner $0, nothing on Modal): per-player passing on pro data

Card: no 29 Oct card by 04:20 UTC -> backup nightly pushed (`triggers/nightly.txt`, free runner). The 28 Oct card's
'worse' rows were already explained and fixed (N0-28c/d). M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before)
The per-player passing figures in `stats.json players[]`: `passes`, `passes_completed`, `passes_forward` /
`_sideways` / `_backward`, `progressive_passes`. In the app they still belong to track pieces, so they are hidden until
M2. This answers: **once M2 gives one id per player, are they right?**

Metrica's 2 pro games, 4 halves, real ids. Truth = Metrica's events by the passer: tried = PASS + BALL LOST that is a
pass (N0-24a's rule), completed = PASS, direction from the ball's start/end, progressive = completed and >= 10 m
forward. Ours = the pipeline on Metrica's positions exactly as run.py does it, perfect ball + our 'streak' / 'flicker'
ball noise (2 seeds) = 20 runs, 40 team-halves, players with >= 10 min. `tools/n0_29b_playerpass.py` ->
`metrica_playerpass.json`.

## Result: counts and the forward/back split hold; a player's completion % does not

| per player | perfect ball | streak | flicker (like our clips) |
|---|---|---|---|
| passes tried: total vs truth / per-player off / order right | 1.02x / 10% / 0.94 | 0.97x / 15% / 0.88 | 1.13x / 16% / 0.86 |
| (baseline 'who was near the ball': order right) | 0.55 | 0.55 | 0.55 |
| passes completed | 1.13x / 17% / 0.95 | 0.99x / 15% / 0.91 | 1.24x / 25% / 0.89 |
| forward passes | 0.89x / 13% / 0.91 | 0.85x / 18% / 0.83 | 0.97x / 16% / 0.86 |
| sideways passes | 1.09x / 20% / 0.75 | 1.11x / 28% / 0.72 | 1.33x / 19% / 0.71 |
| backward passes | 1.24x / 19% / 0.87 | 1.16x / 22% / 0.84 | 1.35x / 25% / 0.81 |
| progressive passes | 1.11x / 14% / 0.87 | 0.86x / 31% / 0.79 | 1.13x / 18% / 0.85 |
| (baseline 'near the ball' for progressive: order right) | 0.10 | 0.10 | 0.10 |
| forward share of his passes, off (pts) - vs giving everyone the team's share | **7.1 vs 12.9** (8/8 better) | 7.9 vs 12.9 (15/16) | 7.1 vs 12.9 (14/16) |
| completion % of his passes, off (pts) - vs giving everyone the team's % | **8.9 vs 6.0** (better 2/8) | 8.1 vs 6.0 (3/16) | 9.1 vs 6.0 (1/16) |

By eye (`players.png`, perfect and flicker ball): passes tried, progressive passes and forward share sit on the
diagonal; completion % is a cloud - players who really complete 40-70% read 75-100%.

So, once M2 exists: **passes, passes completed (as a count), forward/sideways/backward, progressive passes per player
can be shown** (about 1 in 6 off, the busiest players found). **A per-player completion % should not be shown** - it
says less than the team's % does. Shrinking each player's % toward the team's (k = 5-40 passes) does not fix it
(still worse than the team % on the flicker ball). Nothing is shown per player today, so nothing to hide now.

## Side finding: the app's TEAM completion % reads HIGH, not low (corrects N0-24a)
N0-24a's lab built the passes without a possession state; run.py builds them on the spell state
(`IPANEMA_PASS_STATE=spell`), which swallows short spells by the other team - flicker hops, but also real
interceptions. With the app's setting (`tools/n0_29b_teamcompletion.py` -> `metrica_teamcompletion.json`, 40 team-halves),
team completion %, ours minus truth (mean, min..max, points):

| passes built on | perfect ball | streak | flicker |
|---|---|---|---|
| **spell state (the app today)** | **+9.3** (+5.9..+14.7) | +1.8 (-2.6..+6.1) | **+7.7** (+4.5..+12.6) |
| raw possession state | -0.3 | -0.4 | -16.9 |
| no state (what N0-24a measured) | -1.0 | -0.4 | -16.4 |
| no state, hand-backs within 2 s dropped | +4.2 | +4.1 | -5.5 |

So the app's Beta completion % probably reads **~8 points too high** (e.g. 86% shown for a real ~78%). No variant is
better on all three rows (the hand-back rule is closer on perfect and flicker but worse on streak), so **no pipeline
change**. The 'Pass completion %' question under Waiting for Daniel said 'reads ~16 pts low' - corrected to 'reads ~8
pts high'.

## Files
- `tools/n0_29b_playerpass.py` (lab, `--dry` stand-ins), `tools/n0_29b_plot.py` (picture),
  `tools/n0_29b_teamcompletion.py` (team % by pass state), `metrica_playerpass.json`, `metrica_teamcompletion.json`,
  `players.png`.
- `tests/test_n0_29b_playerpass.py` (6): Metrica attempt rules, truth counts with direction / progressive, the lab's
  thresholds equal the app's (3 m, 10 m, completed) and it scores the exported fields, the team-share baseline, the dry
  run, the team % lab. Full suite 760 pass, 4 skipped.
