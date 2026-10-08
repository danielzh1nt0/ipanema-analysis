# C3b: Vallentuna's untrusted camera seconds (8 Oct, worker, free runner + local, $0)

After C3, 30% of Vallentuna's seconds had no trusted camera. Which of them are actually right?

**How:** a blind sample of untrusted rows from the groups a looser rule could let in, plus trusted rows as controls
(tools/c3b_sample.py), drawn on the full-size video on the free runner (tools/c3b_look.py -> c00-c63.jpg). Graded by eye
before opening the key (results/review/c3b_key.json; c00-c09 were seen with their group first, so not blind - they agree
with the rest). Grade: good = boxes / circle / halfway within ~10 px of the painted lines, rough = 10-30 px, off = more.

## Vallentuna (64 pictures, grades.json)

| group | n | good | rough | off |
|---|---|---|---|---|
| trusted (controls) | 8 | 6 | 2 | 0 |
| **'jump from track' anchors** | 16 | **10** | **6** | **0** |
| forward/backward disagree 15-25 px | 20 | 4 | 11 | 5 |
| forward/backward disagree 25-40 px | 12 | 3 | 3 | 5 (+1 can't tell) |
| forward/backward disagree > 100 px | 8 | 1 | 0 | 7 |

'Jump from track' = a fresh anchor placement (every 5 s) more than 60 px from the pose tracked up to it. The pictures say
the fresh placement is right and the tracking had drifted (mostly fast pans). Forward/backward-disagree rows stay out:
a quarter or more are clearly off at every level, so no looser px limit is safe.

## SFK-BP check before switching it on (22 pictures, c3b_sfk/)

16 'jump from track' anchors + 6 trusted controls, blind: **22/22 good** (all within ~10 px).

## Change (on by default)

ipanema/linecal.py: `brave()` trusts 'jump from track' rows (`IPANEMA_TRUST_JUMP=0` = old). Trusted frames (in-play, 2 fps,
results/qa/c3b/coverage.json): Vallentuna 69.7% -> 72.3%, SFK-BP 88.5% -> 91.7%. SFK-BP clip ball keys unchanged (29/34,
B4 284); in the old blank-unsure mode a checked ball comes back (27 -> 28/34, B4 246 -> 264). Tests: tests/test_linecal_brave_refine.py,
tests/test_c3b_sample.py, tests/test_f1_unsure.py. Reaches the app at the next join of each match (Modal, waits for Daniel).

## Not fixed, seen on the way

- The remaining ~28% of Vallentuna are tracked seconds that drift during pans; they need a line fit inside the gap (more
  anchors, e.g. every 2 s instead of 5 s, in a re-calibration run - GPU, not free here), not a looser rule.
- Vallentuna's NEAR touchline is drawn 30-200 px inside the painted white line in ~20 of 64 pictures, also on trusted rows
  (e.g. c29, c39), while the far lines fit. SFK-BP's near touchline fits. Near lines are left out of the fit on purpose
  (lines.py), so the camera base or the pitch size (106 x 64 model) may be a little off for this ground; players near the
  camera may be placed ~1-2 m wrong. Worth a check (C3c).
