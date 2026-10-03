# S3b: does pasting far balls help the ball finder? (3 Oct 2026, free Kaggle, no Modal)

**Status:** training job started on Kaggle's free GPU (3 Oct ~13:40 UTC, about 4.5 h). Results land in
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
