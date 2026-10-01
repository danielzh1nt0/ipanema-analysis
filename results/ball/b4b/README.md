# B4b: spare ball at the goal post (1 Oct 2026)

**Result: new rule works on the spare ball, but it is NOT made the default: two scores dropped a little.**
Option `recur_r=2.0` in `ball.pick_v2` (off by default). Free, local + free runner. No Modal.

The rule (`ball.recurring_spots`): a spot **off the pitch** where something ball-like lies still (half of a 3 s window,
within 1.2 m) in **two or more visits at least 30 s apart** is clutter; off-pitch guesses within 2 m of it are dropped.
Players walking past do not matter. The match ball lying off the pitch before a restart is there only once, so it is kept.
Spots on the pitch (centre spot, corners inside the 1.5 m margin) never count.

On SFK-BP it finds 2 spots, both just behind the left goal line (-4, 34) and (-4, 29).

| SFK-BP (RF-DETR + WASB 30, new players) | off (now) | rule on (2 m) |
|---|---|---|
| picker on the white spare ball (key moments) | 18/18 | **0/18** |
| time the picker sits on the spare-ball spot | 15.8 s | **0.1 s** |
| B4 key, ball moments right | 285/285 | 285/285 |
| match ball before a restart (#240-252) kept | 13/13 | 13/13 |
| 34 moments (Daniel's clicks) | **26/34** | 25/34 |
| who-has-the-ball key (99, possession_simple) | **76/99** | 74/99 |
| dead-ball time / restarts | 19% / 13 | 13% / 9 |
| passes / sequences | 78 / 86 | 90 / 93 |

Picture check where the two pickers disagree (21 s in 12 stretches, 49 moments, sheets `results/free/b4b/old|new/`):
- **During play** (40-42 s, 144-150 s, 155 s, 174-177 s, 296 s; 26 moments): the old picker is on the white spare ball
  at the post 19 times and right once; the new one is on the yellow match ball 18 times, 8 can't tell, none clearly wrong.
- **Stoppage behind the goal** (248-262 s, 23 moments): the match ball went out; the keeper walks to the white ball at the post
  and picks it up for the restart (old sheet #39). The old picker followed that ball; the new one mostly sits on players with
  no ball in view. Daniel's click at 253.8 s is on that white ball -> the one lost of the 34.
- who-has-the-ball: the 2 lost moments are 262.7 s (key: loose, new: dark, end of that stoppage) and 295.7 s
  (key: dark, new ball at a white player's feet next to a dark one).

Why not default: the rule says nothing replaces the current picker unless it scores better, and the 34 moments and the
who-has-the-ball key both drop (by 1 and 2). During play it is clearly better; the losses are around a stoppage where the
spare ball becomes the match ball. Possible next step: drop the spot only while play is on (ball seen moving on the pitch in
the last few seconds), so a ball taken from the post for a restart is followed again.

Also fixed: B4 key moments #29 and #31 (40 s) were the white spare ball, not the match ball (the yellow ball is in play):
key now 285 ball / 18 spare / 9 can't tell.

Files: `tools/b4blab.py` (grading), `tools/b4b_sheet.py` (free runner sheets), `b4blab.json`, `check_moments.json`,
test `tests/test_b4b_recurring.py`. Only calibrated grounds (needs metres); Reymersholm/AIK have no calibration here.
