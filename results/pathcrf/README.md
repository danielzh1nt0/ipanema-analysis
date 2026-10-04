# PC1: possession and passes from player movement only (PathCRF) — 4 Oct 2026

**What:** PathCRF (Kim et al., KDD 2026, MPL-2.0, github.com/hyunsungkim-ds/pathcrf, commit 1b7c6f8) guesses who has the ball from the 22 players' movement alone. It never sees the ball. We used the authors' trained model (trial 140, trained on 7 Bundesliga matches), so Metrica is new data for it. No training, CPU only, about 12 min per match on this machine, $0.

**Answer key:** Metrica's 2 free pro games (4 halves, 1,763 hand-labelled passes), the same key as S1/E5. Passes are matched like in S1: same team, start within 1 s.

**One thing it still needs:** when the ball is in or out of play. Here that comes from Metrica's ball position. In our app it would come from the stoppage detector.

## 1. Perfect player positions (all 22 seen, no error)

| Game / half | Home possession: truth | PathCRF | ours, exact ball | ours, noisy ball | Passes: truth | PathCRF | ours, exact ball |
|---|---|---|---|---|---|---|---|
| 1 / 1 | 55 | 54 | 49 | 44 | 369 | 366 (68% found, 69% real) | 395 |
| 1 / 2 | 57 | 54 | 62 | 68 | 430 | 394 (70% found, 76% real) | 462 |
| 2 / 1 | 51 | 52 | 53 | 56 | 538 | 483 (72% found, 80% real) | 537 |
| 2 / 2 | 54 | 49 | 55 | 33 | 426 | 339 (64% found, 81% real) | 412 |

- **Possession:** PathCRF is off by 1-5 points (average 2.5). Ours is off by 1-6 with the exact ball and 5-21 with the noisy ball. Who-has-the-ball frame by frame matches Metrica 93-95% of the time.
- **Passes:** PathCRF finds 69% of real passes and 77% of its passes are real. Ours with the exact ball (S1) finds 73%, and 73% are real. About the same, but PathCRF uses no ball at all. Its count is 1-20% too low (it misses quick passes). Ours is 3-7% off.
- **Balls lost:** truth 490, PathCRF 414, ours 242 (exact ball) / 62 (noisy ball). The definitions differ a bit, but PathCRF is clearly closer.

## 2. Stand-in for our Veo footage (35 m of the pitch in view around the ball, players out of view filled by straight lines, 0.8 m position error that drifts over ~1 s)

On average 57-62% of the players are in view, about 13 of 22. That is like our follow-cam (F2b: 13-17 seen).

| Game / half | Players seen | Home possession: truth | PathCRF | Agrees frame by frame | Passes: truth | PathCRF |
|---|---|---|---|---|---|---|
| 1 / 1 | 57% | 55 | 52 | 94% | 369 | 458 (64% found, 51% real) |
| 1 / 2 | 59% | 57 | 53 | 93% | 430 | 495 (70% found, 60% real) |
| 2 / 1 | 58% | 51 | 51 | 92% | 538 | 575 (68% found, 63% real) |
| 2 / 2 | 62% | 54 | 48 | 90% | 426 | 423 (62% found, 63% real) |

- **Possession still holds:** off by 0-6 points, 90-94% frame agreement. That beats our noisy-ball possession (off by 5-21) on the same halves.
- **Passes get worse:** 66% found but only 60% real, and the count is 11% too high. Our ball-based passes with S1's "streak" noise found about 68%, with about 70% real. So for passes PathCRF is no better than what we have.
- **Picture** (`pc1_timeline.png`, game 1, first 4 min of each half): position error that is new every frame (a 0.8 m jump every 1/25 s, so impossible speeds) breaks it completely, and the team flips every second. Limited view or drifting error alone barely changes it. It reads speeds and accelerations, so it needs **smooth** tracks.

## Verdict

- **Possession: promising** as a second opinion that needs no ball. It stays within about 5 points where our ball-based possession drifts by 10-20 under noise. **Passes: no gain.**
- **Not wired into the app.** Three blockers on our footage:
  1. It needs one steady identity per player for each stretch of play, for all 22 players. Today our tracks break (M2).
  2. The players out of view must be filled in.
  3. It was trained on pro Bundesliga. There is no youth answer key yet.
- **Next step (PC2, free, local):** once M2 gives one track per player, run it on the SFK-BP clip and grade it on the 99-moment who-has-the-ball key against possession_simple (82/99).

## Files
- `tools/pc1lab.py`: Metrica to PathCRF input, inference, stats, and our ball-based stats on the same frames. Options: `--view`, `--noise`, `--smooth`, `--minutes`.
- `tools/pc1_plot.py`: the timeline picture.
- `tests/test_pc1_pathcrf.py`: 7 tests (no torch needed).
- `pc1_2026-10-04.json`: perfect positions, with truth and ours. `pc1_followcam_2026-10-04.json`: the follow-cam stand-in.
- Needs `torch` (CPU), `torch_geometric`, `pandas`, `scipy`. PathCRF is cloned to /tmp/pathcrf by the tool and not copied into this repo.
- Local test suite: 218 passed, 1 failed (`test_panorama_extend`: numpy 2 on this machine, CI pins numpy<2; unrelated).
