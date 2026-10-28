# N0-28f (28 Oct, worker, local $0, nothing on Modal): per-player average position, thirds and heat map on pro data

Card: the 28 Oct card's 'worse' rows were already explained and fixed by N0-28c/d (frozen copies, the runner's OpenCV);
the 29 Oct card is not out yet. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before)
`stats.json players[]`: `avg_x_m`, `avg_y_m` (average position; also the pass network's node when a player has no pass
spots), `pct_def_third` / `pct_mid_third` / `pct_att_third`, and `stats.heatmaps[id]` (12 x 8 grid per track). The app
hides all of them today (UI contract: per-player maps HIDE, the rest 'not read'), because ids are still track pieces (M2).
They are measured only in frames where the player is **in the picture**, and our camera follows the ball, so the
question is: **once M2 gives one id per player, are they right?**

Metrica's 2 pro games, 4 halves, outfield players (keepers out), every team attacking right. Each moment seen through a
real camera view of our two app clips slid onto Metrica's ball, with our track breaks and jitter (the N0-22e / N0-25b
machinery). 2 games x 2 clips x 2 seeds = 8 runs, ~20 players per half. `tools/n0_28f_playerpos.py` ->
`metrica_playerpos.json`. A player is in the picture about 45% of the time (median).

## Result

| rule | average spot off (median / p90) | thirds off | heat map wrong | who is deepest (rank) | better than seen (pos / thirds / heat, of 8 runs) |
|---|---|---|---|---|---|
| **seen only, one id per player (today's rule after M2)** | **3.2 m / 6.1 m** | 4.5 pts | 15.9% | 0.99 | - |
| today's app: each track piece on its own | 19.2 m / 35.5 m | 30.4 pts | 93% | - | 0 / 0 / 0 |
| last seen spot held 5 s | 2.0 / 4.8 m | 3.7 | 13.0% | 0.99 | 8 / 8 / 8 |
| **held 10 s (new option)** | **1.7 / 4.2 m** | **3.5** | **13.9%** | 0.99 | **8 / 8 / 8** |
| held 20 s | 1.5 / 3.5 m | 3.5 | 14.9% | 0.99 | 8 / 8 / 6 |
| held 60 s | 1.5 / 3.5 m | 3.4 | 15.9% | 0.99 | 8 / 8 / 3 |

- With one id per player the numbers are already usable: the order of the players (who plays deepest / highest) is right
  (0.99), the average spot is ~3 m off and leans slightly toward the ball's side (x -1.2 m, y -2.2 m).
- Keeping a player's last seen spot for a few seconds after he leaves the picture halves the error. 10 s is the one value
  that is better on every run for all three (position, thirds, heat map); longer helps position a bit more but blurs the
  heat map.
- Today, with broken track pieces, the per-'player' numbers say nothing about the player (a piece is a few seconds, ~19 m
  from that player's half-time average). Hiding them in the app is right.

By eye (`avgpos_g2h1.png`, game 2 first half): truth (black), seen only (red x), held 10 s (blue +). Both show the
same formation; the blue crosses sit on or next to the black dots, the red ones are pulled toward the middle and the
near side (e.g. home #8, #5, #4; away #15, #19).

## The app clips (`clips.json`, `tools/n0_28f_clips.py`)
With hold 10 s on today's pieces: 273 / 309 pieces of 2 s or more (SFK-BP / AIK), median seen 11 / 8 s, each piece gets
~10 s of held spots and its average moves 2.5-2.8 m (p90 7.5 m) - toward where it vanished, usually the picture's edge.
On Metrica's pieces the hold makes them slightly worse (19.2 -> 19.7 m). So the hold only pays once M2 joins the pieces.

## Change: option built, OFF (no change to the app or the export)
`analytics.held_track_points` + `IPANEMA_PLAYER_HOLD_S` (default 0 = exactly the old numbers). For M2: switch it on with
`IPANEMA_PLAYER_HOLD_S=10` once players are joined - it feeds avg_x_m / avg_y_m, the thirds and stats.heatmaps;
time_visible_s and everything else still count seen frames only. Nothing for Daniel to decide.

## Files
- `tools/n0_28f_playerpos.py` (lab, `--dry` stand-ins), `tools/n0_28f_clips.py`, `tools/n0_28f_plot.py`.
- `metrica_playerpos.json` (summary + 8 runs + the example half), `clips.json`, `avgpos_g2h1.png`.
- `tests/test_n0_28f_playerpos.py` (6): hold and drop after hold_s, empty frame / goal line clear, default off = old
  average, hold moves it toward the last seen spot (thirds + heat map follow, visible time unchanged), env switch, lab
  scores + dry run.
