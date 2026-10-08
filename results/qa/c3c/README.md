# C3c: Vallentuna's near touchline drawn inside the paint (8 Oct, worker, free runner + local, $0)

**Answer: the pitch is about 65 m wide, not 64.** The camera base is fine. Effect on players is small (~0.7 m at the
near touchline), so nothing is switched on; a fix needs a re-fit of the match rows (Waiting / later).

## How

- 40 Vallentuna + 20 SFK-BP (control) raw frames on the free runner, picked from trusted seconds where the drawn near
  touchline crosses >= 500 px of the lower picture (tools/c3c_frames.py -> frames_vall/, frames_sfk/).
- The near line in shade is sky-blue and wide, so the usual white-line finder misses it; new `paint_mask` in
  tools/c3c_nearline.py catches it (checked by eye on c29/c39/c24).
- **Where the paint lands** (tools/c3c_nearline.py, nearline_frames_*.json): each frame's pose sends the painted near
  line back onto the pitch. Model says 64.00 m.
  - SFK-BP: **64.00 m** (20 frames, IQR 63.98-64.01), 3 px off - fits.
  - Vallentuna: **64.67 m** (37 frames, IQR 64.39-64.90), 41 px off, a bit more near the camera.
- **Which explanation fits** (tools/c3c_fit.py): every frame's pose re-fitted under each hypothesis, cost = mean px
  distance of model lines to paint (far lines / near line). Vallentuna on the 26 frames whose rows fit the far paint
  (fit_clean.json):

| hypothesis | far lines | near line | near line miss (median px) |
|---|---|---|---|
| rows as they are (106 x 64) | 2.93 | 5.60 | 17 |
| re-fit, 106 x 64 | 2.42 | 3.22 | 6 |
| re-fit, 106 x 64.5 | 2.26 | 1.84 | 3 |
| **re-fit, 106 x 65** | **2.24** | **1.07** | **0** |
| re-fit, 106 x 66 | 2.29 | 3.56 | 8 |
| re-fit, 105 x 65 | 2.31 | 1.06 | 0 |
| camera 1 m closer (64 wide) | 2.44 | 1.12 | 0 |
| camera 0.5 / 1 m higher | 2.81 / 3.98 | 2.43 / 1.53 | 4 / 2 |

  65 m wide is the only change that improves far AND near lines; length can't be told (105 = 106). SFK-BP control:
  106 x 64 is best (far 1.87, near 0.66) and every change makes it worse (fit.json, fit_fine.json) - the test finds the
  right answer where we know it.
- By eye (look/vall_n*.jpg, yellow = today's rows, magenta = 105 x 65 re-fit): on the left-end views the magenta near
  line sits on the paint (n05, n15, n25, n35) where yellow runs inside; far lines unchanged.

## What it means

- Players near the camera: the paint is 0.67 m outside where the model puts the touchline, so near-side players are
  placed ~0.5-1 m wrong across the pitch; widths/distances ~1.5% short. Small next to the other errors.
- Fix (not done, no gain worth a paid run now): re-fit Vallentuna's trusted rows with a 106 x 65 pitch on the free
  runner (start from each row, small search, like c3c_fit.refit), then a join (Modal) -> needs Daniel's go. Pitch size
  per ground would go in calibration/<match>_lines_match.json ("pitch": [106, 65]) and linecal.calibration_for_clip.

## Seen on the way (worth a check)

- 12 of the 40 picked Vallentuna rows are TRUSTED but do not fit the far paint at all (old far cost >= 9, e.g. n10,
  n20: lines drawn in the wrong place). Picking views that show the near line favours such wrong poses; C3b's random
  trusted controls were 6 good / 2 rough / 0 off, so they are rare overall, but the near-line pick found them easily.
