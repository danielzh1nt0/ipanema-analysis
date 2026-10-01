# B4c: spare-ball rule only while play is on (1 Oct 2026)

**Result: better by eye during play and nothing lost on the ball keys, but who-has-the-ball drops by one (74 -> 73/99),
so it is NOT the default. Decision for Daniel.** Free: local + one free-runner picture job. No Modal.

The B4b rule (drop guesses at an off-pitch spot where a ball lies still in 2+ visits, e.g. the white spare ball at the left
post of SFK-BP) now only acts **while play is on**: a ball seen moving smoothly at least 2 m inside the lines
(no jumps, at least 1 m in 0.3 s) in the last 5 s. When play stops, a ball at that spot is followed again
(the keeper takes the spare ball from the post for the goal kick at 248-256 s).
Options in `ball.pick_v2`: `recur_r=2.0, recur_play_s=5.0, recur_move_m=1.0, recur_inside_m=2.0`. Off by default.

Graded on the **exact app inputs** of both clips (`picker_inputs.pkl`, as `tools/picktune.py`) with today's tuned picker.

| | no rule (app now) | B4b rule always | **B4c: only while play is on** |
|---|---|---|---|
| SFK-BP 34 moments (Daniel's clicks) | 29/34 | 29/34 | 29/34 |
| AIK 39-ball key | 30/39 | 30/39 | 30/39 |
| B4 key ball moments / restart ball kept | 284/285, 13/13 | same | same |
| picker on the spare ball during play (B4 key) | 11/11 | 0/11 | **2/11** |
| keeper's restart ball followed, 248-256 s (B4 key) | 6/7 | 0/7 | **6/7** |
| who-has-the-ball key (99) | **74/99** | 73/99 | 73/99 |
| dead-ball time / restarts / passes / sequences | 21% / 13 / 87 / 81 | 16% / 12 / 91 / 85 | 19% / 12 / 91 / 81 |

The B4c pick differs from the app's only in 7.9 s: 39.6-42.2 s, 144-149 s (the spare ball during play) and 295.5-295.9 s.
The stoppage is now exactly as before.

**Picture check** (free runner, `results/free/b4c/old|new/sheet_00.jpg`, 18 moments, old and new pick side by side):
- app picker (old): on the white spare ball at the post 16/18, empty grass 1, a player's back with no ball 1. Never on the match ball.
- B4c (new): on the yellow match ball 8/18 (#0, #2, #4-9 at feet or in a duel), can't tell 6, off the ball 4
  (#10 ball ~1 m from the keeper's foot, #11 and #16 empty grass, #12 just beside a dark player's ball).

**The lost who-has-the-ball moment** is 295.7 s (key: dark). The old pick there is on a dark player's back with no ball in
view (#17 old), so "dark" was right by luck; the new pick is at a white player's feet in a duel with a dark player (#17 new),
can't tell from the picture. Same moment B4b lost.

Grid (look-back 3-5 s, speed 0.6-1.5 m per 0.3 s, 1.5 m outside or 2 m inside the lines) in `b4clab.json`: no setting
keeps 74 on who-has-the-ball while acting during play. Slower look-backs or speed limits also start dropping the restart
ball (a person carrying a ball along the touchline during the stoppage counts as play unless the ball must be 2 m inside).

Earlier in the day (offline SFK-BP setup, old picker weights) the same idea gave 26/34 and who-has 74 vs 76; the app
inputs and the 1 Oct picker tuning changed those, so the table above is the one to use.

Rule check: it ties both ball keys and loses 1 who-has-the-ball moment, so it is not made the default.

Files: `tools/b4clab.py` (grading), `tools/b4c_sheet.py` (free-runner sheets), `b4clab.json`, `check_moments.json`,
test `tests/test_b4c_play_on.py`.
