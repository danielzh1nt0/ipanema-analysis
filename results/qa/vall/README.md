# Vallentuna (full match, uploaded through the app 3 Oct) - status 4 Oct evening, all free work

| Part | Before today | Now | Evidence |
|---|---|---|---|
| Teams | 3 vs 11 (red merged into black) | 4 vs 5 per frame on the server; offline with the sun fix 4.3 / 3.8 / 2.4 neither | results/qa/f1c, kits.py min_share + hue rescue |
| Goals / shots | none (zip 0 bytes) | Veo's 22 clips: 17 shots, 5 goals (60:04, 62:22, 64:00, 91:53, 93:13 -> 3-2) | reference/veo_highlights_<match>.txt |
| Match time | 7:15-48:10, 53:40-99:26 | full time 94:05 (teams line up after the 3-2 restart; dog on the empty pitch at 98:44) | results/kaggle/vall_goals_fine, periods/<match>.json |
| Camera | 43% of frames | 70% (doubtful rows accepted when fwd/bwd agree, 16/16 by eye); sliding across longer gaps rejected (9/16 wrong) | results/qa/c3 |
| Ball finder | not measured here | top guess is the ball in 29 of 38 in-play moments, 7 unclear (spare balls by the fence / benches / goal, yellow match ball in deep shade), 2 missed | reference/<match>/ball_gt.json, results/kaggle/vall_ball_key |

Reading: the finder SEES the ball on this pitch about as well as on SFK-BP. The app's ball is near a player in only 47% of
frames, so the loss is after the finder (picker choosing spare balls, or the end-of-match minutes). Measuring the app's own
ball at the 29 key moments needs the exported frames from the server (a read, < 1 cent) - waiting for Daniel's go together
with the join (~EUR 0.10) that puts the goals/shots and the corrected full time into the app.
Players: the sun fix needs a GPU re-track (~EUR 5) - not done (no paid runs).
