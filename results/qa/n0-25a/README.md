# N0-25a (25 Oct, worker, local + free runner $0, nothing on Modal): the app's runs (sprints)

Card: no 25 Oct card yet (00:20 UTC, before 03:00); the 24 Oct card says nothing worse than 23 Oct.
M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

The app's Runs layer (`stats.metrics.runs`: every sprint of 5.5 m/s or more for 0.8 s or longer, with start/end
spot) had a code test (N0-18b) but had never been checked against a truth or by eye.

## Finding: the layer was nearly empty

`metrics.runs` read the speed from **raw** frame-to-frame feet positions. Our feet jitter about 0.3 m a frame, so on
the two app clips 30% of frame speeds are above 5.5 m/s and 14% above 10 m/s (the cap that throws out id jumps). Every
real sprint was chopped into bits shorter than 0.8 s: **SFK-BP clip 2 runs in 5 min, AIK clip 0.**

## Fix (on by default, `IPANEMA_RUNS_SMOOTH=0` = old)

- speed from positions smoothed over 0.8 s within each seen stretch (the same smoothing the distance figures use since N0-22c);
- a raw jump of more than 2 m in 3 frames or less (the track moved to another person) breaks a run - on SFK-BP a
  track zig-zagging between two players was a "sprint";
- rows the tracker filled in with a straight line (`fill_gaps`, marked 'filled', up to 8 m/s) are not seen running.

## Metrica pro data (truth = full tracking, 25 fps)

`tools/n0_25a_runs.py`: 22 players x 20 min per game, each track cut into seen stretches and gaps drawn from our app
clips, our jitter added, with and without id swaps at our rate (25% of gaps). Truth sprint = 5.5 m/s for 0.8 s on the
full track. Found = one of ours on that player overlaps it; real = one of ours overlaps a truth sprint of the player it
actually carries. Both games, 2 seeds (`summary.json`, `metrica_game1/2.json`):

| rule | no swaps: ours / truth | found | real | with swaps: ours / truth | found | real |
|---|---|---|---|---|---|---|
| old (raw speed) | 0% | 0% | - | 0% | 0% | - |
| **new** | **50%** | **42%** | **100%** | **43%** | **30%** | **100%** |
| no jump rule | 64% | 52% | 100% | 55% | 34% | 100% |
| join dips < 0.5 s | 71% | 59% | 96% | 63% | 42% | 95% |
| min 0.6 s | 63% | 50% | 99% | 56% | 34% | 98% |
| smooth 0.4 s / 1.2 s | 55% / 41% | 41% / 36% | 100% | 48% / 36% | 28% / 27% | 100% |
| bridge gaps up to 1 s | 56% | 43% | 92% | 48% | 30% | 91% |

Every run we show is real, but we see about half of them: a sprint needs 0.8 s of unbroken tracking and our seen
stretches are short (median 0.6 s). The jump rule costs ~10 points here because the simulated jitter is rougher than
ours (steps > 2 m: 2.4% simulated vs 1.4% on our clips); on our clips it removes fakes, so it stays. Joining dips
doubles the clip counts (26 -> 49) with fakes the Metrica set does not have -> lab only (`merge_s`, off).

## On our footage, by eye

Every run on both app clips (new rule 37, old rule's own 2) cut as picture sheets on the free runner
(`tools/n0_25a_moments.py`, `tools/n0_25a_frames.py` -> `results/free/n0-25a/`), ids shuffled, graded blind on crops
around the ring (`grades.json`), then opened the key:

| clip | runs old -> new | new: sprinting | not | can't tell |
|---|---|---|---|---|
| SFK-BP (5 min) | 2 -> 26 | 21 | 3 | 2 |
| AIK (5 min) | 0 -> 11 | 11 | 0 | 0 |

New: **32 of 37 real (86%; 32 of 35 clear ones = 91%)**. The 3 wrong are far-side players walking or jogging (far
side: a pixel is many centimetres, so jitter still reads fast). Old rule's 2: 1 real, 1 a player standing next to
the referee. Track pictures (`runs_*.png`: grey feet, red run) show the same.

Tests: `tests/test_n0_25a_runs.py` (7), old rule pinned in `tests/test_metrics_shots_runs.py`. Reaches the app at the next
run (no Modal needed for the code; the app numbers change when a match is re-exported).
