# N0-25d (25 Oct, worker, local only, $0, nothing on Modal)

**Card 25 Oct:** worse than 24 Oct = nothing. M2 / PC2 skipped (main session / waits for M2).

**What was looked at:** the app's **possession over time** and **field tilt over time** charts. They read
`stats.metrics.tilt_windows`: one row per 15 s with `possession_A` (A's share of the frames where someone controls the
ball), `tilt_A` (A's share of the frames controlled in a final third) and `control_s`. The app (Lovable spec, section 10)
draws a **5-minute rolling mean** of those windows (null windows skipped, every window counts the same) and 5-minute bars
("SFK 58% · 0-5'"). Nobody had checked these against a truth before.

## How
Metrica's 2 free pro games, 4 halves (`tools/n0_25d_tilt_windows.py` -> `metrica_tilt_windows.json`, summary
`tools/n0_25d_summary.py` -> `summary.json`). Truth = per frame the team of the last on-ball event, with dead time (ball
out or a foul, until the restart) nobody's. Ours = the pipeline on Metrica's positions with a perfect ball and our two
ball-noise models ('streak', and 'flicker' = the ball dot hopping to a nearby opponent like on our clips), 2 seeds each.
Same 15-s grid for both. "Right team on top" = the curve is on the right side of 50% wherever the truth is at least
10 points from 50%.

## Result: the chart holds

| what the app shows | perfect ball | streak noise | flicker noise (like our clips) |
|---|---|---|---|
| 5-min possession curve, median error | 2.9-3.8 pts | 3.7-5.4 | 3.8-7.5 |
| same, worst 10% of moments | 9-15 pts | 9-15 | 10-17 |
| right team on top | 372/377 (99%) | 745/754 (99%) | 736/754 (98%) |
| 5-min bars, median error | 1.2-4.9 pts | 2.4-5.1 | 2.9-8.1 |
| 5-min bars, right team on top | 23/23 | 44/46 | 46/46 |
| 5-min field tilt curve, median error | 4.4-7.1 pts | 5.9-11.0 | 4.2-10.4 |
| field tilt, right team on top | 489/506 (97%) | 935/1020 (92%) | 966/1020 (95%) |

(ranges = over the halves and seeds; 4 / 8 / 8 runs)

- A **single 15-s window** is not to be trusted on its own (median off 2-20 pts, worst 10% off 19-50 pts), so the app
  must keep smoothing. It does, per the spec.
- Attack direction right in every run.

Picture `metrica_possession_curve.png` (game 1): truth black, ours with a perfect ball blue, ours with flicker red. The
curves follow the truth's swings. Two places where they don't: in half 1 at 27-29 min the truth jumps to 85% home
(a short home spell straight after a 2.7-min stoppage); ours has the stoppage too and keeps ~55%. In half 2 at 36-38 min
the truth dips to ~41% and ours stays at 60-70% with both balls.

## Tried, not switched on
Lab: weight the 5-min curve by `control_s`, so a window with 1 s of control doesn't count as much as one with 15 s
(needs a Lovable change). Better with the perfect ball and flicker (flicker median 3.8-7.5 -> 2.4-5.6 pts, right side
736 -> 746/754, better on 6 of 8 runs), **worse with streak noise** (1 of 8 better, right side 745 -> 727). Also tried in
the pipeline: null windows with < 1 / 2 / 3 / 5 s of control - worse with streak noise (2 s: right side 745 -> 629). Not
better on all rows -> no change. On our own exports it hardly matters anyway: our clips' windows have a median 9-11 s of
control out of 15 (the picker always leaves a ball), only 1-5 windows per match under 3 s, and the weighted curve sits
a median 0.9 pts from the app's (SFK-BP and Vallentuna repo copies).

## Found along the way (lab truth, not app)
N0-24b's field tilt truth (`tools/n0_24b_hightov.truth_tilt`) did not stop the clock at a foul, so a stoppage after a foul
was given to the team of the last on-ball event (game 1 half 1, 25:39-28:19, an injury after a foul, all 'Away').
Fixed in this lab's truth (`is_foul`). Re-done for N0-24b: its truth tilt moves by 0-4 pts on 3 of 8 team-halves
(`n0_24b_truth_fouls.json`) - its conclusion (field tilt holds within ~5 pts) stands. The nightly's possession truth
(`metricalab.truth`) only credits on-ball events until the next event, so it was not affected.

## No pipeline change
`tests/test_n0_25d_tilt_windows.py` (6): the windows pool controlled frames and report `control_s`; empty windows are null;
the app's rolling mean counts every window the same (lab weights); bars vs pooled truth; identical input scores 0; a foul
is dead time in the truth.
