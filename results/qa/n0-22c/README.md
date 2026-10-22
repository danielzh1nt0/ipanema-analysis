# N0-22c (22 Oct, worker, local only, $0, nothing on Modal)

**Nightly card:** 22 Oct card, worse than 21 Oct = nothing.
**Queue before this:** M2 (main session), PC2 (waits for M2), BALL-EDGE (needs a Kaggle GPU; this worker runs only local + the free runner) - skipped, left open.

## Finding
The player numbers in `stats.json` (per player `distance_m`, `high_intensity_m`, `top_speed_ms`; the app shows the team
total "distance, visible players") had values but no test. `analytics.stats` smoothed each track's positions with ONE
0.8 s rolling mean over all its rows, gaps included. Our tracks are full of gaps (2,236 gaps of > 3 frames on the two app
clips, median 0.33 s; seen runs median 0.6 s), so positions from after a gap leaked into the frames before it:

- a player standing still on both sides of a 2 s gap "ran" 4.8 m, all of it at sprint speed (top 6.25 m/s);
- the real movement during a gap was not counted at all (gaps > 3 frames were skipped).

Pictures (`track_sfkbp.png`, `track_aik.png`, grey = tracked feet, red = old smoothing, blue = new, orange = not seen):
by eye the red line goes flat through each gap and bends towards the next run's spot before the gap starts - clearly
wrong on both; the blue line follows the dots.

## Fix (on by default; `IPANEMA_RUN_SMOOTH=0 IPANEMA_GAP_LINE_S=0` = old numbers)
`analytics.track_motion` (new, used by `stats`): positions smoothed per seen run; a gap of at most 2.5 s (the track
stitching window) adds the straight line across it when that is 0.5-7 m/s (`IPANEMA_GAP_MAX_MS`, about a U15 sprint;
faster = another person); it is high-intensity when > 5.5 m/s; gap speeds never count as the top speed.

## How it was chosen (no answer key for our players -> Metrica pro tracking as the truth)
`tools/n0_22c_metrica.py`: 22 Metrica players x 10 min (25 fps), each cut into seen runs and gaps drawn from our two
app clips, our measured jitter added, and - because on our clips a fifth of the gap jumps are faster than 10 m/s
(another person, not a run) - id swaps to a team-mate after 25% of gaps, which reproduces that share (22-23%).
Total distance / high-intensity distance as % of the truth, and the median per-player distance error:

| Metrica game | swaps | old | **new (per run + gap line <= 7 m/s)** | per run only | gap line up to 10 m/s |
|---|---|---|---|---|---|
| 1 | none | 87-88% / 147-151% / 12-14% | **93% / 103-106% / 7-8%** | 74% / 79% / 26% | 96% / 129% / 5% |
| 1 | 25% | 84-85% / 139-146% / 21-22% | **90-92% / 111-116% / 14%** | 75-77% / 90% / 28% | 94-96% / 143-151% / 11-12% |
| 2 | none | 87-88% / 277-294% / 12-13% | **92-94% / 153-163% / 7-8%** | 73-74% / 94-105% / 26% | 95-96% / 206-218% / 5% |
| 2 | 25% | 81-83% / 216-237% / 22-24% | **89-90% / 150-163% / 16-18%** | 72-75% / 90-106% / 29-32% | 92-93% / 215-238% / 12-14% |

New is better than old on all three numbers in every row (`metrica_game1.json`, `metrica_game2.json`). Caps 6 / 8 m/s
bracket 7 (in the JSON as line_cap6 / line_cap8). Also tried, not kept: a gap line only between seen runs of at least
0.1-0.5 s (against lone detections) - Metrica distance fell from 95-96% to 79-87% of the truth, and on our clips the
gaps next to short runs are not more often impossible than the others (> 10 m/s: SFK-BP 22% / 27% / 29% for runs
<= 0.07 s / 0.07-0.3 s / >= 0.3 s, AIK 20% / 19% / 18%).

## Real app clips (`tools/n0_22c_run_smooth.py`, exact picker inputs, `app_clips.json`)
| clip | team distance A old -> new | B | high-intensity A | B | median top speed |
|---|---|---|---|---|---|
| SFK-BP 5 min | 5,775 -> 5,940 m | 4,810 -> 4,898 | 1,236 -> 993 | 938 -> 840 | 9.15 -> 8.27 m/s |
| AIK 5 min | 4,366 -> 4,702 m | 3,856 -> 4,250 | 736 -> 705 | 713 -> 715 | 8.68 -> 7.89 m/s |

Team distance +2-10%, as the Metrica check predicts (old under-counted by ~15%).

## Limits (not fixed here)
- A gap line can still join two different people when the jump is plausible (AIK track 646, 190-201 s: lone
  detections between gaps add ~37 m; SFK-BP track 824 at 275 s: a 12 m jump in 2 s counted as a sprint). One track per
  player (M2) is the real fix.
- High-intensity distance and top speed are still too high (top speeds 8+ m/s on most tracks = jitter); both stay
  hidden in the app. The centred mean also shortens each run's ends a little (6% on a 6 s run; old too).
- Reaches the app at the next run / join (Modal, Daniel).

## Tests
`tests/test_run_smooth.py` (8): still across a gap = 0 m (old 4.8 m at 6.25 m/s); moved during a gap = straight line, not
a sprint; gap limits (10+ m/s, 3 s gap, standing); no-gap tracks unchanged; runs never mixed; old switch through
`stats`; both tools' dry runs.
Suite here: **553 passed**, 0 failed.
