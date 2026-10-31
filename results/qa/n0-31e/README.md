# N0-31e (31 Oct, worker, free runner + local, $0, nothing on Modal): final-third entries skip balls off the pitch

31 Oct card: nothing worse than 30 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## Why
The app's 'Where did they get in?' (final-third entries, `stats.field.<team>.final_third_entries[].y_m`) drew some
entries off the pitch: on the repo's app copies 34 of 687 entries had the ball more than 2 m past a touchline (AIK up
to 488 m on a 64 m wide pitch). A ball in the air projects past the far line; a pick on the fence or a spare ball does too.

## What changed
- `ipanema/metrics.py` `field_tilt_and_entries(..., W=)`: a ball more than `ENTRY_MARGIN_M` past a line is skipped as
  if not seen - it makes no entry and does not count for field tilt. `IPANEMA_ENTRY_ONPITCH=0` = old rule.
- Margin **4 m** (`IPANEMA_ENTRY_MARGIN_M`, first try was 2 m) - set from the blind check below.

## Blind check by eye (42 picture sheets, free runner, run 2 at the right video times)
21 entries the 2 m rule drops, mixed with 21 it keeps (Vallentuna 37, SFK-BP 5). Question per sheet: at the entry,
is the ring on the ball and the ball on the pitch? Graded on zoomed crops (`zoom/z00-z10.jpg`) before opening the key
(`grades_by_eye.json`, `moments_key.json`).

| | ball on pitch (Y) | not (N) | can't tell |
|---|---|---|---|
| dropped by the 2 m rule | 7 | 12 | 2 |
| kept | 11 | 5 | 5 |

- The 12 right drops: ball in someone's hands past the line (4), white spare balls by the fence (5), ring in the
  hedge / by a board (2), a throw-in held overhead (1).
- The 7 wrong drops are all balls **on the far touchline** read 2.2-3.6 m out (one 7.8 m): far away a pixel is metres.
  The real off-pitch balls read 2.4-29 m out, mostly 4+.
- With 4 m: of the 19 decided drops it keeps 6 of the 7 good balls and still drops 9 of the 12 bad ones.
  Wrong calls 7 (2 m) -> **4** (4 m); old rule (no drop) 12. Small sample and the margin was picked on it - noted.
- The 5 kept entries graded N are the ring on empty grass or on a boot in mid-pitch: a ball-finder miss, not this rule.

## Numbers
- Metrica pro data (`metrica_entries.json`, `margin4/metrica_entries.json`): found / real / count / tilt identical
  for old, 2 m and 4 m with every ball model (perfect ball 72% found, 96% real); the off-pitch entry with the streak
  ball goes 1 -> 0. No cost.
- App copies rebuilt (`margin4/clips.json`): entries old -> 4 m: Vallentuna A 89 -> 88, B 130 -> 124; SFK-BP B 42 -> 41;
  AIK and the SFK-BP clip unchanged; field tilt unchanged (within 1 pt). 2 m had dropped 21, 4 m drops 10.
- Reaches the app at the next run (no Modal now).

## Files
`tools/n0_31e_entries.py` (Metrica), `tools/n0_31e_clips.py` (app copies + blind sheet list), `tools/n0_31e_frames.py`
(free runner sheets -> `results/free/n0_31e/`), `tests/test_n0_31e_entries_onpitch.py` (9 tests).
`run1_wrong_times/` = run 1 grades on sheets cut at the wrong times, not used.
