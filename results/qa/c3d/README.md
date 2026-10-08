# C3d: wrong TRUSTED camera rows on Vallentuna (8 Oct, worker, free runner + local, $0)

**Answer: about 1 trusted in-play second in 25 is clearly wrong on Vallentuna (~150 of 3,413, rough estimate), and
about 1 in 3 is 10-30 px off. No picture check tried here can veto the wrong ones without also throwing out good
ones, so nothing is switched on.** SFK-BP (the control) has essentially none.

## How

- Free runner read both full videos once, front to back, and scored every posed second (tools/c3d_farcost.py,
  13-14 min per match): **far cost** = mean px distance from the drawn far lines (everything except the near touchline)
  to the painted white lines, 1280x720, capped at 15 px (same measure as C3c; reproduces C3c's numbers on its 40 frames).
- Trusted in-play seconds:

| far cost (px) | 0-3 | 3-5 | 5-7 | 7-9 | 9-12 | 12-15 |
|---|---|---|---|---|---|---|
| Vallentuna (3,413 s) | 1,254 | 732 | 536 | 378 | 247 | 266 |
| SFK-BP (4,241 s) | 4,238 | 3 | 0 | 0 | 0 | 0 |

- 60 Vallentuna seconds picked ~10 per cost band, drawn with thin model lines, neutral ids, **graded blind** before
  opening the key (vall/look/, vall/grades.json): g = lines on the paint, r = 10-30 px off, o = clearly wrong, ? = can't tell.

| band | g | r | o | ? |
|---|---|---|---|---|
| 0-3 | 7 | 2 | 0 | 1 |
| 3-5 | 4 | 4 | 1 | 1 |
| 5-7 | 6 | 4 | 0 | 0 |
| 7-9 | 7 | 3 | 0 | 0 |
| 9-12 | 4 | 3 | 3 | 0 |
| 12-15 | 2 | 7 | 0 | 1 |

  Weighted by band: ~56% good, ~33% rough, **~4% clearly wrong** (4 seen, so anywhere ~1-10%), ~7% can't tell.

## Why the cost can't veto them (vall_raw/, veto.json, tools/c3d_veto.py)

- On Vallentuna a high far cost mostly means **long shadows**: the white-line finder misses paint in the shade, so
  right poses score 11-14 px (x004, x010, x028) while a wrong one scores 4.4 (x023). A 9 px limit catches only 2 of the
  4 wrong rows and drops ~510 seconds, most of them good or rough.
- Shade-aware paint (C3c's paint_mask) separates good from rough much better (good mostly < 2.2 px), but it also picks
  up the **blue extra lines** on this pitch: best limit (>= 6) catches 3/4 wrong, drops 2/30 good and 8/23 rough.
- "Move on re-fit" (how far the lines jump when the pose is re-fitted to the paint): the wrong poses are off by tens of
  metres, a small local re-fit can't find the right view (x023, x057 move only 11-40 px), and right-end views with blue
  lines move 20-30 px the wrong way (refit pulled onto blue lines). No clean limit.
- The 4 wrong rows are whole-view mistakes (wrong pan along the pitch / a box drawn where there is none), at both
  ends; the C3c corner views (n20, n24: camera turned into the right corner, centre drawn) are the same kind.

## What it means

- Wrong trusted seconds are rare and ground-specific (sun + shadows + blue lines); SFK-BP's rows sit on the paint.
- A cheap paint check is not the fix; a re-calibration of Vallentuna (better anchors, 106 x 65 from C3c, blue lines
  masked) is. Not worth a paid run now. Tool kept for other grounds: `run` gives a per-second cost in ~14 min free.
