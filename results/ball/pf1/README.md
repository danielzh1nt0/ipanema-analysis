# PF1: particle-filter ball picker vs today's picker (3 Oct 2026, worker, local, $0, no Modal)

**Result: no gain. Today's picker (Viterbi, `ball.pick_v2`) stays.** The particle filter is kept as an option only
(`ipanema/pfball.py`, not wired into the pipeline).

## What was built
- `ipanema/pfball.py` `pick_pf`: 600 guesses ("particles") of where the ball is, each in one of three modes:
  **ground** (rolls on, off-pitch unlikely), **air** (flies freely, off-pitch fine), **with player** (stuck to a tracked
  player's feet, where the finder often misses it). Camera pan removed between frames like the current picker. Weighted by
  the same finder guesses and the same static-clutter rule as the current picker (shared code: `ball.v2_rows`,
  `ball.drop_static`; the current picker's output is unchanged, checked: 30 / 29 / 284). A few fresh particles go on strong
  guesses every frame so a lost ball is found again. The answer for a frame is read 15 frames later (smoothing) and snapped
  to the guess with most particle support.
- `tools/pf1lab.py`: runs settings on the exact app inputs of both clips, grades all keys + the 28 graded passes.
- `tests/test_pf_picker.py`: follows a rolling ball through missed frames, ignores strong one-frame decoys, same seed =
  same answer. Suite: 203 passed; the 4 failures are this machine missing torch / supervision and one old numpy issue,
  same without my change.

## Scores (keys: AIK 39, SFK-BP 34, B4 285; passes: 20 real / 8 fake graded)
| picker | AIK | SFK-BP 34 | B4 | real passes kept | fake passes kept |
|---|---|---|---|---|---|
| **today (Viterbi)** | **30** | **29** | **284** | **20** | 8 |
| particle filter, first settings | 25 | 24 | 269 | 12 | 2 |
| best settings (3 seeds) | 26-28 | 26-27 | 279-283 | 11-12 | 0-1 |
| best, ground mode only (no air / player) | 26 | 26 | 276 | 10 | 0 |
| best, no smoothing (lag 0) | 25 | 25 | 255 | 12 | 4 |

27 settings in `sweep.json` (smoothing 0/8/15/30 frames, injection 0.5-10%, 600-1,500 particles, guess spread, confidence
weight, mode speeds, seeds). Seeds alone move the score by about 1 on each key.

## Why it loses
- The Viterbi picker looks at the whole clip at once and picks the single best path. The particle filter only looks
  15 frames ahead; with a longer look (30) it gets worse because fresh particles have no real past.
- When the ball is lost and re-found far away, the filter needs several frames to move over; the Viterbi path jumps at once.
- The three modes do help a little (ground-only is 0-4 worse), but not enough.

## The one interesting bit
It keeps **0-1 of the 8 fake passes** (today 8), because it does not flick between still guesses near the feet. But it also
loses 8-10 of the 20 real passes (it drops the ball more often in fast play), so the pass count is not better either.
The fake-pass problem stays where S8/B10 put it: the finder cannot tell a ball at a foot from a mark by the line.
