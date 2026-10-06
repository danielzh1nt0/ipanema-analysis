# Passes: what's wrong and what fixes it (6 Oct)

## Answer key
Daniel tapped every pass in the Vallentuna clip at 66:40–68:40: **21 passes**.

## Our current method (track the ball and players, then count changes of the player on the ball)
- Counts **52 passes**. 16 of them are real, so 31% of what we count is a real pass.
- Rebuilt offline from the match's own tracking data (`tools/passlab_taps.py`), and tried 108 settings of the counting rule. The best setting gives 41 passes with 15 real. The counting rule is not the problem.
- The causes are upstream:
  - The ball dot jumps, by up to 80 m in 1.5 s.
  - During a dribble, "the player on the ball" flips between teammates who stand close together.
  - In two moments the team colours were wrong.

## Research
- **SoccerNet Ball Action Spotting** (CVPR challenge, 2023–25) spots 12 ball actions straight from the video, among them PASS, DRIVE, HIGH PASS, CROSS, SHOT and THROW IN. It is trained on broadcast EFL video and reaches about 0.6 mAP@1 s.
- **T-DEED** (CVsports 2024, GPL-3.0 code) publishes a checkpoint for that challenge.
- **dude.k** (2025) is a refined T-DEED. No public weights.
- **FOOTPASS** (2026) adds who passed (team and shirt number). Its videos are under NDA.
- **PathCRF** (2026) detects events from player tracks only, with no ball: F1 0.76 on full 22-player tracking. We see only about 7 players per team, so it is a weaker fit for us.

## Test: T-DEED on our footage, unchanged, no training (Kaggle, free)
Same clip, model passes = PASS + HIGH PASS + CROSS + FREE KICK, merged when less than 0.8 s apart, a match counts within 1.5 s.

| model threshold | model passes | real passes found (of 21) | model passes that are real |
|---|---|---|---|
| 0.2 | 39 | 19 | 49% |
| **0.3** | **23** | **16** | **70%** |
| 0.4 | 19 | 13 | 68% |
| current method | 52 | 16 | 31% |

At 0.3 the model's count is about right (23 vs 21) and more than twice as precise as our current method, with no work on our side. It does not say which team passed. That comes from our tracking: the team of the player nearest the ball at the moment of the kick.

## Next
1. Daniel taps the other 3 clips (SFK–BP, SFK–AIK, Vallentuna 20:00) to check that this holds on other matches and pitches. The model's output for those clips is already in `results/kaggle/tdeed_passes/tdeed/`.
2. Add the team from our tracking and score again.
3. If it holds: run the model on the 3 full matches (Kaggle, free) and use its passes in the export, which needs one join per match on Modal. Later, fine-tune on our own taps.
4. The licence is GPL-3.0: fine for running on our own server, but check before shipping the code to customers.

## Which team passed (6 Oct afternoon)
Score: the 16 model passes that match a tap, counting how often the team is right.

| method | team right |
|---|---|
| Team head of the SoccerNet 2025 team model (left/right in the picture) | 9 of 14, near chance. Its passes were also slightly worse (12–14 of 21 found) |
| Our app's "who has the ball" field around the kick | 10 |
| Our nearest player to the exported ball | 8–9 |
| Player nearest our raw ball candidates at the kick (best of 36 settings, so optimistic) | 11 |

The cause is the same as before. On this second-half clip in low sun our ball choice is often wrong (the ball picked on a spectator at the fence), and some players have the wrong team colour.
- **"When is a pass?"** is solved well enough: the model gives 23 passes against 21 real.
- **"Which team?"** is not solved: about 65% right.

**Running now:** the model on the full playing time of all 3 matches (Kaggle, free, about 3 h).

## All 4 clips: our count vs the model (2 minutes each, model threshold 0.3)
| clip | our current method | model | real (tapped) |
|---|---|---|---|
| SFK–BP 25:00 | 23 | 13 | – |
| SFK–AIK 25:00 | 28 | 25 | – |
| SFK–Vallentuna 20:00 | 36 | 14 | – |
| SFK–Vallentuna 66:40 | 52 | 23 | 21 |

- Our method is close to the model on the BP and AIK clips (dark against white kits, even light) and 2–2.5× too high on both Vallentuna clips (red against black, hard sun).
- So the inflation is mainly a Vallentuna problem. That fits the colour and ball issues seen there.

## The fix that works: our passes, confirmed by the model
- Keep only those of our passes that the model also sees (within 0.7 s, model score >= 0.2, one-to-one). Our pass keeps its own team, players and positions, so the pass map works.
- Vallentuna 66:40 clip: our method alone gave 52 passes with 16 real. Confirmed by the model: **22 passes, 14–15 of them real, team right 12–13 of 14–15 (about 85%)**. The real number is 21.
- In the pipeline: `analytics.confirm_passes`, used in `run.analyse` when `overrides/<match>_kicks.json` exists (env `IPANEMA_PASS_CONFIRM`; tolerance `IPANEMA_PASS_CONFIRM_TOL`). The kicks file is made from the full-match model run with `tools/make_kicks.py`.
- Caveat: tuned on one tapped clip of 21 passes.
