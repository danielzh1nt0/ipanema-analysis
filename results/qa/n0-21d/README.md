# N0-21d (21 Oct, worker, local $0, nothing on Modal)

**Card:** results/nightly/2026-10-21.md says nothing worse than 20 Oct (tests 147/147, ball 22/34, all player checks ran).

**What was looked at:** the part of the app step `run.analyse` that grades the ball and sums up a match. While checking
it on a small full match with a half-time break, possession during the break came out as one team's.

**Fault (fixed):** in a full match, frames outside match time (before kick-off, half-time, after the whistle) and
frames with an untrusted camera are emptied on purpose (no players, no ball). The possession model keeps the last
state when there is no ball, so **the team that had the ball at the whistle got the whole break** - in possession %,
possession seconds and the possession field of every exported frame (the app's timeline).

App copies of the full matches in the repo (`python tools/play_state_check.py`, every 3rd frame, output in
`app_copies.txt`):

| match | app shows | match time only | where it went |
|---|---|---|---|
| SFK-BP full first half (possession shown in the app) | A 45 / B 55 | **A 53 / B 47** | 51:00-55:00 after the whistle (coach's time) all given to B: 2,397 frames |
| Vallentuna full match (possession withheld in the app) | A 36 / B 64 | A 43 / B 57 | half-time + after full time given to B: 6,502 frames |
| AIK full first half | A 35 / B 65 | can't tell | the repo copy has only 2 of 12 frame files |

So on SFK-BP the app has the wrong team on top. No picture needed: the frames in question are emptied by design
(0 players, no ball); the half's end at 51:00 is the coach's, in periods/SFKBP1109.json.

**Fix:** `possession.out_of_play` sets those frames to 'loose' (nobody's, not counted in possession %), applied in
`run.analyse` right after the possession model when the match has a play mask (full matches only; 5-min clips have
none and are unchanged). Loose % and dead % in the summary are now of match time, not of the whole recording.
`IPANEMA_PLAY_STATE=0` = old.

**Test:** `tests/test_play_state.py` (3): the unit rule, and a 40-s full match (A on the ball except 10-15 s, B on the
ball at the half-time whistle, break 15-25 s) through the real `analyse` + export: new A 84 / B 16 (truth 25 s / 5 s =
83 / 17), break frames have no possession; the old switch gives A 63 / B 37 and B through the break.
Full suite: 526 pass, 4 skipped.

Reaches the app at the next full-match join (Modal, CPU, cents) - under "Waiting for Daniel".
