# N0-31c (31 Oct, worker, free runner + local, $0, nothing on Modal): SFK-BP's half-time, second-half kick-off and full time by eye

31 Oct card: nothing worse than 30 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## Why
`periods/SFKBP1109.json` had only the first-half kick-off checked by eye (555 s). The other three boundaries came
from the coach's notes or from Veo's goal times. AIK's periods were wrong twice (D6: 3 min of empty pitch counted
as play; N0-30b: match end off), so SFK-BP's were looked at the same way.

## How
`tools/n0_31c_sfk_periods.py` on the free runner (whole video downloaded, 3.6 GB, 6164 s): frames every 10 s around
each boundary, then a second run every 5 s from half-time to 3615 s. Sheets: `results/free/n0_31c/sheet_ht.jpg`,
`sheet_ko2.jpg`, `sheet_ft.jpg`, `run2/sheet_ko2b.jpg`.

## Result
| boundary | file said | by eye | verdict |
|---|---|---|---|
| first-half end | 3060 s (coach 51:00) | play at 3050, players walking off at 3070, pitch empty from ~3100 | **right** (within 10 s) |
| second-half kick-off | 3733 s (from Veo goal times, +-30 s) | players back out ~3450, both teams lined up at the centre 3505, play from 3515; play on in every frame 3515-3830 | **wrong: 3510 s** (3 min 43 s earlier) |
| full time | 5940 s (coach 1:39:00) | attack near the far goal 5910, players walking from ~5920-5930, handshake line-up 5960+ | about right (10-20 s late), kept |

The coach's own note "58:29 when players came back out" was the kick-off itself (58:30 = 3510 s).

## What changed
- `periods/SFKBP1109.json`: second half 3733 -> **3510 s** (note added). The old file left the first 223 s of the
  second half (about 10% of it) out of any full-match run, and clips cut from 3510-3733 s could not get an attack
  direction (run.py places clips by the periods).
- Nothing in the app changes now: the app shows SFK-BP's first half only (cut at 3300 s), and that half's start and
  end are both now confirmed by eye. Reaches the app only with a full-match SFK-BP run.
- Not touched: `tools/f1_check.py` keeps its own old copy of the periods (a finished check).

## Tests
`tests/test_n0_31c_sfk_periods.py` (4): windows cover every boundary, frame cutting on a stand-in video (seek lands
on the right frame, past-the-end skipped), the second-run window spec, and the periods file holds the by-eye values.
Dry runs on a local clip before each free-runner push.
