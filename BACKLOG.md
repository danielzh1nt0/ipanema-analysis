# Ipanema work queue

The scheduled worker takes the first item marked `[ ]`, does it, ticks it `[x]` with a one-line result, commits, and stops.

## Rules (Daniel's, non-negotiable)
- **No Modal spend without Daniel's explicit "go" in chat.** This includes anything under $1. The worker only uses this machine and the free GitHub runner (`triggers/free.txt`, `triggers/playerbench.txt`, `triggers/nightly.txt`).
- Prove offline first. Test before running (a dry run with stand-in models, plus the `tests/` suite). Nothing replaces a better model unless it scores better.
- Every change gets a test. Results go into `results/`, committed.
- If an item needs money or a decision, write it under "Waiting for Daniel" and move on.
- Plain, short language in anything Daniel reads.

## Queue (priority order)
- [ ] P4. TOP: team split fails on Reymersholm night piece (results/review/who_reym_2026-09-28.md): spectators become a team, green+white merge. Fix without calibration: drop people who stay near the frame's top edge / hardly move (bench, spectators); more sample frames; check grass removal at night vs green kits. Test on the free runner (RF-DETR CPU on 24 frames) for Reymersholm, Spånga, SFK; must not break SFK. Then re-run E3 (needs one more GPU tracking run, ~$0.15, ask Daniel).
- [ ] E3. possession_simple (51/61 vs 27/61 on SFK-BP) needs a second ground before it becomes the default: who-has-the-ball strips + answers on Reymersholm or Vasalund. Needs ball picks there (GPU: Kaggle once the token works, or a small Modal run with Daniel's go). Then: passes/turnovers/possession computed from possession_simple, graded on the same keys.
- [x] E2 (28 Sep: key grown to 99 clear moments, 38 of them fresh and blind; possession_simple 31/38 on the fresh ones and 82/99 overall vs 43/99 for the current model; it has neither the dead-ball nor the missing-carrier fault; default waits for E3's second ground. results/review/who_2026-09-28.md) Possession on our footage is right in only 9/24 clear moments (results/review/who_2026-09-28.md). Fix, graded on who_answers.json: (a) dead-ball wrongly called during play (5/24): don't trust off-pitch projections of single ball picks; (b) carrier not found for a player on the ball (7/24): link in pixels (ball near feet) not only metres. Grow the key to 100+ moments (more strips, Reymersholm/Spånga) first so fixes aren't tuned to 24.
- [ ] (PARKED by Daniel 28 Sep: skip Kaggle for now) K1. Kaggle free GPU (about 30 h/week): a notebook pushed by the Kaggle API from the free runner that reads matches from R2, runs WASB (30 peaks) + the click finder, and cuts ball/not-ball crops at the checked labels of Vasalund, Solheim, Spånga, Djursholm (thousands of balls). Then retrain the scorer (T0) and grade on exam + clip. No Modal.
- [ ] (PARKED) K2. Same Kaggle route for training runs (ball finder on RF-DETR, B7) so training stops costing money.
- [ ] H1. Check martinjolif/football-ball-detection (HF, CC BY 4.0, 1,237 images) as extra ball pictures: look at a sample sheet, see if it is broadcast-only; use only if it helps the exam.
- [ ] T0. TOP: second-opinion scorer on WASB guesses. WASB sees the ball in 31/34 clip moments and 79/81 exam frames but ranks it first only 26 and 56 times (results/ball/probe.md). Train a small crop classifier (ball vs not) on CPU: positives = Daniel's 436 clicks, negatives = WASB's other peaks in the same frames (probe.json has the guesses for 142 frames; more from the WASB cache). Re-rank, then picker. Grade: exam 81 + clip 34, cross-validated by frame. Also: stop capping WASB at 6 peaks (keep ~30, local maxima) - needs a WASB re-run on the clip (GPU ~5 min, ask Daniel).
- [x] B1b. Re-run B1 and the miss count on FUSED guesses (both finders), not WASB-only: picklab only had WASB guesses locally. Needs the click-finder guess cache for the clip (see Waiting for Daniel).
- [x] B8. Why the app clip scored 22/34 when the offline fused test scored 25-26/34: compare finder version, stride, far-zoom, cache.
- [x] B2. Look below the cut-off: at the 8 moments with no guess on the ball, do the finders see the ball weakly? Needs raw heatmaps (only on Modal) -> write what is needed under "Waiting for Daniel".
- [ ] P1. Spånga striped kits: check the re-run (results/qa/tracktest_p15u-vs-spanga-2026-09-25) after the track-vote change; if players are still missing, find out why and fix, re-test on all grounds.
- [ ] B3. (LOW, after S3; round 8 collapsed on SoccerTrack and Gemini agrees the fisheye view is too different) Prepare SoccerTrack v2 ball labels in pixels (repo script scripts/calibration/project_tracking_to_image.py, ground balls only, check the known y-flip problems). Free runner, HF_TOKEN secret. Output: crops + labels ready for training, plus a picture sheet to check projection.
- [ ] B4. Grow the ball answer key from 34 to 300+ moments without Daniel clicking: frames where both finders AND the picker agree with high confidence, plus a check sheet Claude grades by eye.
- [ ] E1. (NOW BLOCKS STATS FIXES) Main ball score = "who has the ball" (team, per stretch of play) on our own footage, not only "within 30 px". Start from the 40 who-has-the-ball moments already in results/review/stats_SFKBP1109_s1200 + B4 moments. Report it in the nightly scorecard. (Gemini review 28 Sep, agreed)
- [ ] T1. Second-opinion check on each ball guess: a small 3-frame crop classifier (ball moving vs line/shoe/cone) that removes false guesses before the picker. Train on CPU from existing clicks + guesses; grade on the 34 moments + exam. (Gemini review)
- [ ] S3. Pasted balls: cut real ball crops from Daniel's clicks and paste them (blurred, 4-8 px) onto our own grass frames, far side especially, as extra training pictures. Picture sheet to check before any training. (Gemini suggested white circles; real crops are better)
- [ ] Z1. Picker: "ball in the air" flag, so a ball off the pitch plane isn't punished by pitch-metre rules; grade on the 34 moments.
- [ ] PF1. Particle filter picker (ground / air / with player) vs the current Viterbi, same 34 moments + exam. Only replaces Viterbi if it scores better.
- [ ] PC1. Passes/possession from player movement only (PathCRF, MPL-2.0) on the free Metrica data; compare with our ball-based stats.
- [x] B5 (28 Sep: 8/13 at feet or crowd, 4/13 in the air, 1 open; results/picker/misses.md). Why the finders miss: for each of the 13 missed moments on SFK-BP, crop the frame and sort into: ball in the air / at feet / in a crowd / far away / blurred / off-pitch confusion. Picture sheet + counts in results/picker/misses.md. Decides where training labels must come from.
- [ ] B6. Second ball answer key on another ground (Vasalund or Reymersholm): 30+ moments from Daniel's earlier clicks (results/review/*) so ball numbers are not judged on one clip only. No new clicking.
- [ ] B7. Ball finder on RF-DETR (Apache): write the training script + dataset format (from click labels + B3), dry-run 1 epoch on CPU with 50 images. Real training waits for Daniel's go.
- [ ] P2. Players on all 6 training matches with the free runner (20 s each, 3 spots per match): table of tracked seconds, missed players, referee counted, per ground. Fix the worst ground.
- [ ] P3. Tracklet joining (gta-link idea, MIT): join broken pieces of the same player; measure pieces per 20 s and time tracked on SFK, Spånga, Reymersholm.
- [ ] S1. Stats on pro data: passes still 12-17% off. Find which passes we miss/add (Metrica), fix the rule, keep possession within 7 pts.
- [ ] S2. Set pieces from motion (stoppages_from_motion): score on Metrica, target 80% found with few extras.
- [ ] A1. Prepare the "one paid run" list: exactly what goes to the app (new players + current ball) on which clips, estimated cost, command ready. Put it under Waiting for Daniel.
- [ ] N1. Every morning: read results/nightly/<date>.md, and if anything got worse than the day before, put it at the top of this queue.

## Waiting for Daniel
- (done 28 Sep, blocked by P4) E3 on Reymersholm, two small Modal jobs: [fetch] the existing ball guesses for p15u-vs-reymersholm s1500 (file copy, < 1 cent) + [tracktest-rf:1500:300:p15u-vs-reymersholm-2026-09-18] GPU player tracking (~11 min, ~$0.15). Then strips + answers + possession_simple (player-height ruler, no calibration) are free.
- Kaggle token: the GitHub secret KAGGLE_API_TOKEN is missing or no longer valid (Kaggle says 'authentication required'). Needs a new token from kaggle.com/settings/api saved as secret KAGGLE_API_TOKEN in the GitHub environment MODAL_TOKEN_ID, plus variable KAGGLE_USERNAME. Kaggle account must be phone-verified for GPU + internet. Then re-run: triggers/kaggle.txt = kaggle/smoke.py.
- WASB re-run on the SFK-BP clip keeping ~30 peaks per frame (GPU ~5 min) so T0 and the picker can use the buried guesses.

## Done
- [x] 28 Sep: E1/E2 who-has-the-ball key grown to 61 clear moments (Claude by eye); possession_simple 51/61 vs viterbi 27/61, both key halves agree. Option only. results/review/who_2026-09-28.md
- [x] 28 Sep: S1b turnovers/possession on Metrica: 3 s rule finds 20% of losses; frame-to-frame ball speed made the possession model say 'nobody has it' 76-87% of the time under noise; windowed speed fixes it on pro data but not proven on our clip (no answer key). Options added, defaults unchanged. results/metrica/turnovers_2026-09-28.md
- [x] 28 Sep: T0 first scorer (small 3-frame CNN, 260 ball pictures from the SFK-BP clicks, CPU): exam 61 -> 63-68/81 when blended with the finder score (3 seeds); clip 25 -> 24-25/34 = no gain. Not promoted. Too few ball pictures; next: thousands from the checked labels of 4 matches (needs GPU crops -> Kaggle free GPU, K1). results/ball/scorer/
- [x] 28 Sep: B2 probe (GPU 7.5 min): WASB sees the ball in 31/34 and 79/81, but ranks it first in only 26 and 56; the app kept only 6 WASB peaks. Problem = choosing, not seeing. results/ball/probe.md
- [x] 28 Sep: B8: app 22/34 = round-10 click finder (better on the exam, worse on the clip) + old players. tools/fusedlab.py
- [x] 28 Sep: B1 "ball with this player" option in the picker. No gain: 21/34 at best (tools/posslab.py, results/picker/posslab.json). Of the 9 moments with no finder guess, a player's feet are on the ball in only 3, so the ball is mostly in the air / far / not near a detected player. Option kept, off by default. -> the finder is the limit; B5 and B7 next.
- [x] 28 Sep: new player pipeline (RF-DETR + per-match kits + gap filling + keeper + duplicates): full SFK-BP clip, dark players tracked 1.5 -> 4.1 s, light 0.7 -> 7.0 s.
