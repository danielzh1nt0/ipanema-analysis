# N0-31d (31 Oct, worker, free runner + local, $0, nothing on Modal): AIK's kick-offs and full time by eye

31 Oct card: nothing worse than 30 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## Why
After N0-31c found SFK-BP's second-half kick-off 3 min 43 s off, the same check for AIK. In
`periods/p15u-vs-aik-2026-09-21-bd09.json` only the first-half end (3225 s, D6) had been looked at by eye. The
two kick-offs came from a detector and full time was the end of the video. Earlier side findings (N0-26b, N0-30a,
N0-30b) said all three were off, but had not looked at frames close together. They left it for "the next re-join".

## How
`tools/n0_31d_aik_periods.py` on the free runner (whole video, 4.0 GB, 6899 s). It cut 69 frames, every 5 s around
the kick-offs and every 10 s around half-time and full time. Sheets: `results/free/n0_31d/sheet_ko1.jpg`,
`sheet_ht.jpg`, `sheet_ko2.jpg`, `sheet_ft.jpg`.

## Result
| boundary | file said | by eye | verdict |
|---|---|---|---|
| first-half kick-off | 366 s | players warming up / tying boots until ~415; teams lined up 420-450; captains with the referee at 450; dark SFK at the centre spot 455-460; play from 465 | **wrong: 460 s** (94 s late) |
| first-half end | 3225 s (D6) | play at 3210; players walking at 3220 | right (control) |
| second-half kick-off | 3950 s | players come back out 3940-3990; lined up 3995-4015 with white AIK at the centre spot; play from 4020 | **wrong: 4015 s** (65 s late) |
| full time | 6899 s (video end) | play until ~6720-6730; substitutes in bibs walk on at 6740; handshake line 6760-6770 | **wrong: 6730 s** (169 s early) |

The direction is also right: dark SFK kicks off from the right half in the first half, and white AIK kicks off in
the second half.

## What changed
- `periods/p15u-vs-aik-2026-09-21-bd09.json`: `[[366, 3225], [3950, 6899]]` -> **`[[460, 3225], [4015, 6730]]`**
  (note added). In total, 328 s of warm-up, coin toss, break and after-whistle time had been counted as play.
- **App copy (first half, unchanged until the next AIK join):** the 94 s before the kick-off has 5 'lost the ball',
  5 'won the ball', 5 sequence ends, 2 passes and 16 'better option' events. All of them come from the warm-up and
  coin toss. N0-30a measured that starting at ~440 s moves SFK's possession share by under 1 point (AIK possession
  is withheld anyway).
- Not touched: finished checks keep their own copies of the old values (`tools/f1_check.py`,
  `tools/n0_30a_aik_poss.py`, `tools/n0_30b_aik_h2*.py`).

All three app matches now have every period boundary checked by eye: SFK-BP (N0-31c), Vallentuna (4 Oct) and AIK
(here).

## Tests
`tests/test_n0_31d_aik_periods.py` (3): the windows cover the old and suspected boundaries, a dry run of the tool on a
stand-in video, and a check that the periods file holds the by-eye values. Dry run on a local clip before the push.
Full suite 831 pass. 3 fail only because `supervision` is not installed on this machine (the runner has it).
