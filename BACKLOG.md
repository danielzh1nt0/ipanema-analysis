# Ipanema work queue

The scheduled worker takes the first item marked `[ ]`, does it, ticks it `[x]` with a one-line result, commits, and stops.

## Rules (Daniel's, non-negotiable)
- **No Modal spend without Daniel's explicit "go" in chat.** This includes anything under $1. The worker only uses this machine and the free GitHub runner (`triggers/free.txt`, `triggers/playerbench.txt`, `triggers/nightly.txt`).
- Prove offline first. Test before running (a dry run with stand-in models, plus the `tests/` suite). Nothing replaces a better model unless it scores better.
- Every change gets a test. Results go into `results/`, committed.
- If an item needs money or a decision, write it under "Waiting for Daniel" and move on.
- Plain, short language in anything Daniel reads.

## Queue (priority order)
- [ ] B2. Look below the cut-off: at the 8 moments with no guess on the ball, do the finders see the ball weakly? Needs raw heatmaps (only on Modal) -> write what is needed under "Waiting for Daniel".
- [ ] P1. Spånga striped kits: check the re-run (results/qa/tracktest_p15u-vs-spanga-2026-09-25) after the track-vote change; if players are still missing, find out why and fix, re-test on all grounds.
- [ ] B3. Prepare SoccerTrack v2 ball labels in pixels (repo script scripts/calibration/project_tracking_to_image.py, ground balls only, check the known y-flip problems). Free runner, HF_TOKEN secret. Output: crops + labels ready for training, plus a picture sheet to check projection.
- [ ] B4. Grow the ball answer key from 34 to 300+ moments without Daniel clicking: frames where both finders AND the picker agree with high confidence, plus a check sheet Claude grades by eye.
- [ ] B5. Why the finders miss: for each of the 13 missed moments on SFK-BP, crop the frame and sort into: ball in the air / at feet / in a crowd / far away / blurred / off-pitch confusion. Picture sheet + counts in results/picker/misses.md. Decides where training labels must come from.
- [ ] B6. Second ball answer key on another ground (Vasalund or Reymersholm): 30+ moments from Daniel's earlier clicks (results/review/*) so ball numbers are not judged on one clip only. No new clicking.
- [ ] B7. Ball finder on RF-DETR (Apache): write the training script + dataset format (from click labels + B3), dry-run 1 epoch on CPU with 50 images. Real training waits for Daniel's go.
- [ ] P2. Players on all 6 training matches with the free runner (20 s each, 3 spots per match): table of tracked seconds, missed players, referee counted, per ground. Fix the worst ground.
- [ ] P3. Tracklet joining (gta-link idea, MIT): join broken pieces of the same player; measure pieces per 20 s and time tracked on SFK, Spånga, Reymersholm.
- [ ] S1. Stats on pro data: passes still 12-17% off. Find which passes we miss/add (Metrica), fix the rule, keep possession within 7 pts.
- [ ] S2. Set pieces from motion (stoppages_from_motion): score on Metrica, target 80% found with few extras.
- [ ] A1. Prepare the "one paid run" list: exactly what goes to the app (new players + current ball) on which clips, estimated cost, command ready. Put it under Waiting for Daniel.
- [ ] N1. Every morning: read results/nightly/<date>.md, and if anything got worse than the day before, put it at the top of this queue.

## Waiting for Daniel
(nothing yet)

## Done
- [x] 28 Sep: B1 "ball with this player" option in the picker. No gain: 21/34 at best (tools/posslab.py, results/picker/posslab.json). Of the 9 moments with no finder guess, a player's feet are on the ball in only 3, so the ball is mostly in the air / far / not near a detected player. Option kept, off by default. -> the finder is the limit; B5 and B7 next.
- [x] 28 Sep: new player pipeline (RF-DETR + per-match kits + gap filling + keeper + duplicates): full SFK-BP clip, dark players tracked 1.5 -> 4.1 s, light 0.7 -> 7.0 s.
