# Ipanema work queue

The scheduled worker takes the first item marked `[ ]`, does it, ticks it `[x]` with a one-line result, commits, and stops.

## Rules (Daniel's, non-negotiable)
- **No Modal spend without Daniel's explicit "go" in chat.** This includes anything under $1. The worker only uses this machine and the free GitHub runner (`triggers/free.txt`, `triggers/playerbench.txt`, `triggers/nightly.txt`).
- Prove offline first. Test before running (a dry run with stand-in models, plus the `tests/` suite). Nothing replaces a better model unless it scores better.
- Every change gets a test. Results go into `results/`, committed.
- If an item needs money or a decision, write it under "Waiting for Daniel" and move on.
- Plain, short language in anything Daniel reads.

## Queue (priority order)
- [ ] B1b. Re-run B1 and the miss count on FUSED guesses (both finders), not WASB-only: picklab only had WASB guesses locally. Needs the click-finder guess cache for the clip (see Waiting for Daniel).
- [ ] B8. Why the app clip scored 22/34 when the offline fused test scored 25-26/34: compare finder version, stride, far-zoom, cache.
- [ ] B2. Look below the cut-off: at the 8 moments with no guess on the ball, do the finders see the ball weakly? Needs raw heatmaps (only on Modal) -> write what is needed under "Waiting for Daniel".
- [ ] P1. Spånga striped kits: check the re-run (results/qa/tracktest_p15u-vs-spanga-2026-09-25) after the track-vote change; if players are still missing, find out why and fix, re-test on all grounds.
- [ ] B3. (LOW, after S3; round 8 collapsed on SoccerTrack and Gemini agrees the fisheye view is too different) Prepare SoccerTrack v2 ball labels in pixels (repo script scripts/calibration/project_tracking_to_image.py, ground balls only, check the known y-flip problems). Free runner, HF_TOKEN secret. Output: crops + labels ready for training, plus a picture sheet to check projection.
- [ ] B4. Grow the ball answer key from 34 to 300+ moments without Daniel clicking: frames where both finders AND the picker agree with high confidence, plus a check sheet Claude grades by eye.
- [ ] E1. Main ball score = "who has the ball" (team, per stretch of play) on our own footage, not only "within 30 px". Start from the 40 who-has-the-ball moments already in results/review/stats_SFKBP1109_s1200 + B4 moments. Report it in the nightly scorecard. (Gemini review 28 Sep, agreed)
- [ ] T1. Second-opinion check on each ball guess: a small 3-frame crop classifier (ball moving vs line/shoe/cone) that removes false guesses before the picker. Train on CPU from existing clicks + guesses; grade on the 34 moments + exam. (Gemini review)
- [ ] S3. Pasted balls: cut real ball crops from Daniel's clicks and paste them (blurred, 4-8 px) onto our own grass frames, far side especially, as extra training pictures. Picture sheet to check before any training. (Gemini suggested white circles; real crops are better)
- [ ] Z1. Picker: "ball in the air" flag, so a ball off the pitch plane isn't punished by pitch-metre rules; grade on the 34 moments.
- [ ] PF1. Particle filter picker (ground / air / with player) vs the current Viterbi, same 34 moments + exam. Only replaces Viterbi if it scores better.
- [ ] PC1. Passes/possession from player movement only (PathCRF, MPL-2.0) on the free Metrica data; compare with our ball-based stats.
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
- Copy the click-finder guess cache for the SFK-BP clip (cache/SFKBP1109_s1200/ball_cands_clicks*.pkl) from the Modal volume into the repo. A file read, no GPU; cost near zero, but it is Modal. Unblocks B1b and B8.
- B2 (weak guesses below the cut-off on the missed moments): about a minute of GPU.

## Done
- [x] 28 Sep: B1 "ball with this player" option in the picker. No gain: 21/34 at best (tools/posslab.py, results/picker/posslab.json). Of the 9 moments with no finder guess, a player's feet are on the ball in only 3, so the ball is mostly in the air / far / not near a detected player. Option kept, off by default. -> the finder is the limit; B5 and B7 next.
- [x] 28 Sep: new player pipeline (RF-DETR + per-match kits + gap filling + keeper + duplicates): full SFK-BP clip, dark players tracked 1.5 -> 4.1 s, light 0.7 -> 7.0 s.
