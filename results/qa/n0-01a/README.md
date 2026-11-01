# N0-01a (1 Nov, worker, local, $0, nothing on Modal): event subtitles in plain words

No 1 Nov card yet (00:20 UTC, before the 03:00 backup time); 31 Oct card: nothing worse than 30 Oct. M2 (main session)
and PC2 (waits for M2) skipped.

## Why
Each event in `match_data.events[]` has a `subtitle` the app can show under the title. The text was built with plain
f-strings, so a missing value printed as Python's `None` or a bare dash:

- `A sequence ended · pass intercepted | 0.87 s · gain None m` (9 of 14 sequence ends on the SFK-BP clip)
- `A lost the ball | press — s · — within 5 m · not regained` (most lost balls: no press inside 8 s)
- `B won the ball | forward pass — s · -3.1 m in 5 s` ("minus 3 m in 5 s" = the ball went back)
- `throw-in · —` (restart team unknown)

On the repo's 4 app copies 763 of 3,066 event texts had one of these (SFK-BP full 205, clip 10, AIK 223, Vallentuna 325);
lost / won balls are 'Beta' events, so these are the ones the app lists.

## What changed (`ipanema/export.py`)
`lost_subtitle`, `won_subtitle`, `pass_subtitle`, `sequence_subtitle`: a missing value drops its part or says what did not
happen; numbers rounded; 'back' instead of a minus. Payloads (the numbers the app reads) are unchanged.

| before | after |
|---|---|
| press — s · — within 5 m · not regained | not pressed within 8 s · not won back |
| press 2.87 s · 1 within 5 m · not regained | pressed after 2.87 s · 1 opponent within 5 m at 2 s · not won back |
| forward pass — s · -3.1 m in 5 s | no forward pass within 8 s · ball 3.1 m back in 5 s |
| forward pass 3.0 s · 0.0 m in 5 s | forward pass after 3 s · ball no further forward in 5 s |
| 0.87 s · gain None m | 0.9 s |
| 19.0 m, gain 16.6 m | 19 m · gain 16.6 m |
| throw-in · — | throw-in |

Rebuilt from each event's own payload on the 4 app copies: texts with None / nan / bare dash 763 -> **0**
(`subtitles.json`, more examples there). Better-option, shot, goal and high-turnover texts unchanged.
Reaches the app at the next run (no Modal now).

## Files
`tools/n0_01a_subtitles.py`, `tests/test_n0_01a_subtitles.py` (4 tests).

Side note (not changed): the UI contract (results/app/ui_contract_2026-10-02.md) lists `ball_m_before_press` on lost balls
as a position for a turnover map; it is a distance (metres the ball moved between the win and the first press), so it
can't place a dot.
