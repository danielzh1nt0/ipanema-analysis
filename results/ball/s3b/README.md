# S3b: does pasting far balls help the ball finder? (3 Oct 2026, free Kaggle, no Modal)

**Status:** training job started on Kaggle's free GPU (3 Oct 13:09 UTC, about 4.5 h; GitHub run 37125155685). Results land in
`results/kaggle/ballfinder_paste/` by themselves; the next worker run grades them.

## What the job does
- Same training as B7 (the app's ball finder, exam 84/108): same matches, crops, seed, 12 epochs. No extra negatives,
  no at-feet copies, so the only change is the pasting.
- Into every training crop that has a labelled real ball, ONE far ball is pasted (4-8 px, sized from the players in that
  frame, light and codec matched; `ipanema/pasteball.py`). The crop is moved (real ball still inside) so a far spot fits.
  The pasted ball gets a label like a real far ball. Crops without a ball stay empty, so the finder is never taught to
  ignore a real ball.
- Balls to paste: 84 clean cut-outs of checked balls (SFK-BP training clicks outside the test clip and away from exam
  frames, + 4 more grounds). Never Reymersholm/kitprobe frames.
- Person boxes from the pipeline's detector, so balls are never pasted on a player.
- Also writes `paste_sheet.jpg` (48 pasted balls, 3x zoom) to check by eye.

## How to grade (next run)
1. Look at `results/kaggle/ballfinder_paste/paste_sheet.jpg`: balls on grass, not on people/lines' tops, size sensible.
2. `result.json`: exam (B7 84/108, ball found 74/81), clip 34 (top guess B7 26).
3. `PYTHONPATH=. python tools/newfinder_grade.py results/kaggle/ballfinder_paste`: through the picker, new vs old
   finder from the same job: 34-key (29), B4 key (284), graded passes.
4. Promote only if better on the keys, not within noise (earlier retrains swing +-3 on the exam).

## Checks done before starting
- `tests/test_pasteball.py`: 2 new tests (pasted ball stays inside the crop and away from the real ball; test clip
  balls are never used as paste sources). Whole suite: 207 passed.
- Dry run on this machine (CPU, stand-in videos made from our frames, tiny model, 1 epoch): data, pasting, training,
  weights, exam, clip and candidate steps all ran. `dryrun_paste_sheet.jpg` = its pasted balls (stand-in player
  boxes, so placement there is not meaningful).
- On real frames with real player boxes, a far spot fits in the crop 143 of 150 times (95%), ball 6-8 px.

## Result (graded 3 Oct, worker, local $0) - NOT promoted
Job finished on free Kaggle in 242 min (train 103 min, 4,847 of 4,983 ball crops got a pasted ball: 1,918 open grass,
2,409 near feet, 520 near a line; 5-8 px).

**Paste sheet by eye:** the pasted balls sit on grass, never on a player or the top of a line, size looks right for the
far side. Many look like real far balls; some are a bit too clean / too round, and most sit alone on open grass.

| | B7 (app) | pasted |
|---|---|---|
| Exam at 0.25 (108 frames) | 84 | 81 |
| Exam: ball found (81) | 74 | 68 |
| Exam: no-ball frames right (27) | 10 | 13 |
| Exam: top guess on the ball (81) | 70 | 70 |
| Clip 34, finder alone top guess | 26 | **29** |
| B4 key (285), finder alone top guess, same job | 284 | 256 |
| Through the picker (today's weights): 34 key / B4 | 29 / 284 | 26 / 282 |
| Through the picker, best of 18 weight settings | 30 / 282 | 27 / 280 |
| Graded passes real / fake kept (same job old 15 / 4) | 15 / 4 | 13 / 3 |

- The finder alone ranks the ball first more often on the 34 clicked moments (29 vs 26), but loses everywhere else:
  fewer balls found on the exam, 28 fewer top guesses on the B4 key, and 3 fewer moments once the picker runs.
  Re-tuning the picker for it does not close the gap (`tools/s3b_grade.py`, `grade.json`).
- B4 leans to the old finder (it was built where the old finder and WASB agreed), so 256 vs 284 overstates the loss;
  the picker result on Daniel's 34 clicks (26 vs 29) is the fair number and it still loses.
- Likely why: the pasted balls teach "small round blob on open grass"; on real footage the hard balls are at feet, in
  crowds and in the air (B5), so pasting moves the finder's confidence the wrong way for the picker.
- Weights kept in `results/kaggle/ballfinder_paste/` for reference. B7 stays in the app. Pasting is closed unless
  the pasted balls can be made to look real (S3: still spotted 43/48 blind).
