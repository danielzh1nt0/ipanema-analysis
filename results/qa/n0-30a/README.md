# N0-30a (30 Oct, worker, local only, $0, nothing on Modal): AIK possession vs a blind by-eye sample

**Nightly card:** no 30 Oct card yet (00:20 UTC, before 03:00, so no backup push); the 29 Oct card says worse than 28 Oct = nothing.
**Queue before this:** M2 (main session) and PC2 (waits for M2) skipped, left open.

**What was looked at:** the 25 Oct accuracy status lists AIK's whole-match possession (Kaggle join, `results/kaggle/full_match_52d795-join`:
46% SFK) as "not trusted yet - needs a blind by-eye possession sample". 60 random first-half frames of that match were already
cut on Kaggle (`results/kaggle/aik_poss_sample/poss`, seed 7, full size, nothing drawn) but nobody had graded them.

## How
- Each of the 60 frames graded by eye: who has the ball, dark SFK (= team A in the export, checked on `kit_A.png`) or white AIK (= B),
  or can't tell. Full frame + a 2.2x zoom around the exported ball spot (the zoom shows no team or possession, but it is centred on our
  ball, so it is blind to our answer, not to our ball). Restarts count for the team taking them (throw-in, corner, goal kick), marked dead.
- Grades: `aik_poss_grades.json`. Comparison: `tools/n0_30a_aik_poss.py` -> `result.json` (export row nearest each moment, and the
  majority of the export rows within +-1 s). Picture: `decided_moments.jpg` (red label = export disagrees within +-1 s).

## Result
| | by eye: SFK share | 95% range | export at the same moments | export row agrees (nearest / +-1 s) |
|---|---|---|---|---|
| all 27 decided moments | **30%** (8/27) | 16-48% | 41% (of 17 it decides) | **14/17** / 20/24 |
| 14 sure ones | 29% (4/14) | 12-55% | 45% | 9/11 / 10/12 |
| 22 in play (restarts out) | 32% (7/22) | 16-53% | 46% | 10/13 / 15/19 |

- The export's own **first half reads 39% SFK** - inside the by-eye range (16-48%). So the sample does **not** show the AIK
  possession to be wrong; it leans to SFK having the ball less (30% vs 39%), but 27 moments can't tell 9 points apart.
  The 5 Oct look (8 decided frames, ~25% SFK) points the same way.
- Moment by moment the export is mostly right where it says something (14 of 17, 82%). The 3 misses are the raw state's flicker:
  the export row says A, the rows 0.3 s either side say B (white player clearly on the ball in p13, p23, p41).
- 33 of 60 frames can't be decided by eye (ball not visible, in the air, set pieces forming, players walking after a stop) -
  a 60-frame sample only gives ~27 answers. To tell 30% from 39% needs ~150-200 more decided moments.
- The whole-match 46% includes the second half, which this sample does not cover - not checked here.

## Side finding: the halves start before the kick-off (not changed)
Frame p00 (400.5 s) shows only white players warming up and p01 (430.5 s) the captains' coin toss, but `periods/p15u-vs-aik-2026-09-21-bd09.json`
starts the first half at 366 s; N0-26b put the kick-off at 428-446 s, and the second-half kick-off at 3984-4005 s (file: 3950 s). In those
~75 s + ~40 s the export gives the ball to AIK almost every row (366-430 s: 96% B). Starting the halves at 440 / 3990 s moves the export's SFK
share by under 1 point (first half 39.4 -> 40.3%, whole match 45.5 -> 46.1%), so it is not worth a change on its own; worth fixing with
the next AIK re-join.

## Tests
`tests/test_n0_30a_aik_poss.py` (4): nearest row / +-1 s majority, the counts and agreement, the 95% range, and that the committed grades
cover all 60 frames, hold no pipeline answer and match the frame times. Full suite 786 pass, 7 skipped.

No new decision: the status doc's "show or withhold AIK possession" stays open, but this sample gives no reason to withhold
the first half (it fits), and no evidence either way on the second half. $0, nothing on Modal.
