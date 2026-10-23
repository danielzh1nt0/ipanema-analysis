# N0-23b (23 Oct, worker, local + free runner, $0, nothing on Modal)

**Card:** no 23 Oct card at 04:20 UTC -> backup nightly pushed (`triggers/nightly.txt`); the card came in: worse than 22 Oct = nothing.
M2 / PC2 / BALL-EDGE skipped (main session / waits for M2 / needs a Kaggle GPU; this worker runs only local + the free runner).

**What was looked at:** the counter-press figures in `stats.json` teams[]: `time_to_press_median_s` (how long until the
team that lost the ball has a player within 2 m of the new carrier), `pressed_within_2s_pct`, `near_at_2s_median`
(losers within 5 m of the carrier 2 s after the win), and the per-event `time_to_press` / `near_at_2s` the app lists on
each ball-won event. Nobody had checked them.

## How it was checked (no clicking)
- Truth = Metrica's free pro games (2 games x 2 halves): every labelled BALL LOST -> the winner's RECOVERY (or first
  on-ball event); with Metrica's true positions, the carrier = the winner's player nearest the true ball (<= 2.5 m, the
  app's rule); first frame a loser is within 2 m of him, until the winner's chain ends or 8 s (`tools/n0_23b_press.py`).
- Ours = the app's own steps on Metrica's positions (`tools/metricalab.ours`, turnovers after `confirm_losses`), with a
  perfect ball, our 'streak' noise (players thinned to ~65%, ball misses / wrong picks) and 'flicker' ball noise, 2 seeds.
- New (`--view`): players seen through our app clips' REAL camera views (SFK-BP and AIK, slid onto Metrica's ball) plus
  our clips' detection gaps and jitter - ~43% of all players seen, the ones near the ball mostly kept.
- Per lost ball: our kept turnovers matched to truth wins (same winner, within 2 s).

## Findings - the figure holds, no change
| Metrica, 4 halves | 'pressed within 2 s' same as truth | truth yes / ours yes | time to press within 0.5 s | median time error | near at 2 s same |
|---|---|---|---|---|---|
| perfect ball, all players | **133/153 (87%)** | 92 / 92 | 87/120 | 0.00 s | 69/90 |
| flicker ball | 191/238 (80%) | 125 / 132 | 81/161 | 0.00 s | 84/119 |
| streak (players ~65%) | 35/49 (71%) | 29 / 29 | 13/39 | +0.08 s | 5/11 |
| perfect ball, our camera views | 128/158 (81%) | 97 / **81** | 63/110 | 0.00 s | 45/81 |
| flicker ball, our camera views | 102/130 (78%) | 70 / 68 | 44/83 | 0.00 s | 27/52 |

Picture: `press_vs_truth.png`. Rows: `metrica_press.json`, `metrica_press_view.json`, `summary.json`.

1. **Per lost ball the yes/no is right 78-87% of the time**, no bias in the median time (0.0 s), and the same number of
   'yes' as the truth on the same balls. This is much better than the other transition figures (N0-22f regain, N0-23a
   forward pass), because pressure needs only the players near the ball, not a clean possession flip.
2. **Through our camera** the presser is sometimes not in the picture or in a detection gap: with a perfect ball we say
   'yes' on 81 where the truth has 97 (~1 in 6 presses missed); 'near at 2 s' then reads lower (0.48 vs 0.87 with
   flicker). So on our footage the % reads a little low and `near_at_2s_median` (0 or 1) is rough.
3. **Half-level numbers are not comparable with Metrica's 'all losses'**: the pro truth over all labelled losses is
   50-65% within 2 s, median 0.0 s - most labelled losses are duels, pressed at the moment of the win. The app counts only
   kept losses (S9 drops duels and hand-backs), so its half % is 31-64% and median 0.0-2.2 s. On the SAME lost balls it
   matches (point 1). Not a bug - the app's tooltip should say "lost balls (duels excluded)", like the S9 label.
4. **Tried and rejected:** starting the press clock at the winner's first touch (like N0-23a's forward pass) - per-event
   agreement drops 133 -> 127 (perfect) and 191 -> 183 (flicker), 'yes' over-counted 152 vs 125. Kept as today (clock from
   the frame the winner controls a slow ball). `metrica_press_from_touch_rejected.json`; no code switch left in.

## Our app clips (repo copies, unchanged)
SFK-BP clip: A 11 losses, press 1.3 s, 36% within 2 s, near 0; B 9, 0.32 s, 33%, near 1. AIK clip: A 63 losses, 2.7 s,
21%; B 70, 3.29 s, 14%. Plausible for youth play next to the pros' 50-65% (pros incl. duels); the lower AIK figure fits
point 2 (more of the pitch out of view) - not provable by eye on picture sheets (the ball ring is the limit, N0-23a).

## Change
- No change to the pipeline. New: `tools/n0_23b_press.py` (truth + `--view` camera thinning), `tools/n0_23b_plot.py`,
  `tests/test_n0_23b_press.py` (4). Suite 574 pass (local, with supervision installed).
- For the app (no paid run needed): keep 'pressed within 2 s' and 'time to press' visible; 'near at 2 s' is rough on
  our footage - fine as a per-event detail, not as a headline.
