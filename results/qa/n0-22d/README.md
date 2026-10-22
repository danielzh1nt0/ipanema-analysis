# N0-22d (22 Oct 12:20 UTC, worker, local, $0, nothing on Modal)

**Card 22 Oct:** worse than 21 Oct = nothing (tests 152/152, ball 22/34, pro-data passes 15% / possession 8 pts off).
M2 / PC2 / BALL-EDGE skipped (main session / waits for M2 / needs Kaggle).

**What was looked at:** coverage of the app code after N0-22c (suite 553 pass). `metrics.py` is 100% covered, but one
of its outputs had an untested case: `metrics.tilt_windows`, the 15-s windows behind the app's "possession over time"
and "field tilt over time" charts (Lovable spec, results/app/lovable_fixes_2026-10-04.md section 10).

## Fault (fixed)
A window in which **neither team had the ball** gave `possession_A = 0.0`. That reads as "the opponent had 100%".
`tilt_A` already said `null` in the same case.
- These windows are before kick-off, at half-time and after the whistle. Since N0-21d those frames are nobody's
  possession, so every such window gets 0.0.
- Repo copies, with N0-21d applied as the next join will export them (`tools/n0_22d_tilt.py`): SFK-BP full first half
  53 of 222 windows, Vallentuna 73 of 401. All of them are outside the match periods. None are inside.
- The app's chart (5-min rolling mean, null windows skipped): **if** the app filters the periods as the spec says, nothing
  changes. If it doesn't, or does it loosely at the edges, the first and last 2.5 min of each half are pulled towards the
  opponent: up to 92 pts on SFK-BP, 33 pts on Vallentuna. The half-time break would also show the opponent at 100%.
  Picture: `possession_over_time.png`, red = old, blue = new, drawn without the period filter.
- AIK is left out because the repo copy has only 2 of 12 frame files.

## Fix (on by default; `IPANEMA_TILT_EMPTY=zero` = old)
`possession_A` is now `null` when the window has no controlled frame. Each window also gets `control_s` (seconds of
control the share rests on), so a reader can weight thin windows or skip them. A real 0.0 (only B had the ball) stays 0.0.
Windows with control are unchanged (checked on all windows of both copies).

Not changed: in-period windows with only a few seconds of control. Dropping windows under 1-5 s of control did not
make the 5-min curve closer to the true weighted possession on SFK-BP or Vallentuna (mean error 1-2 pts either way).

**Tests:** `tests/test_tilt_empty.py` (4):
- an empty window becomes null, with `control_s`
- a window without ball metres becomes null, and a real 0.0 stays 0.0
- the old switch
- a 50-s full match through the real `analyse` + export: the break window is null and the played windows are not. The
  old rule gives 0.0 there.

Full suite: 557 pass, 4 skipped (no torch here).

Reaches the app at the next run. No app change is needed: the spec already says to skip null windows.
