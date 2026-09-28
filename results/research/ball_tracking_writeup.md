# Ipanema: ball tracking. Full situation report (28 Sep 2026)

*Written for an outside reviewer (e.g. Gemini). Everything here is measured unless marked "not measured" or "I don't know".*

---

## 1. What we are building and why the ball matters

Ipanema turns youth/amateur football match video into stats and clips for coaches. The first customer is Sollentuna FK. The target is roughly 90% accuracy on players, ball, pitch lines and stats, shown in a web app (built in Lovable).

The ball is now the main thing holding back accuracy. Nearly every stat depends on it:
- possession %
- passes (completed/attempted)
- turnovers (lost/won)
- set pieces / dead-ball time
- attacking sequences
- "who has the ball" at any moment

Player tracking was improved a lot on 28 Sep (see section 9). On the Metrica pro data used as an answer key, our stats code is the smaller error source. With realistic ball noise, pass counts are 12–17% off, and most of that comes from ball errors.

## 2. The footage

- **Source:** Veo cameras. The only thing we can download is Veo's **follow-cam** (a virtual camera that pans/zooms to follow play), 1920×1080 at ~29.97 fps. The raw 180° panorama is not available to us.
- **Level:** youth and amateur matches (e.g. P15/P09 youth teams) on grass and artificial pitches around Stockholm, day and night (floodlights).
- **What the ball looks like:**
  - Far side: often **4–6 px** across, blurred when kicked, and it merges with white pitch lines, shoes, socks and spare balls lying around the pitch.
  - Near side: ~10–20 px.
  - Behind players it disappears entirely.
- **Camera:** the follow-cam moves constantly. We estimate a homography (pixels → pitch metres) for every frame from pitch lines (our own line model + calibration). It is good on the main clip and weaker on grounds without a calibration.
- **Matches available:**
  - One main labelled match, "SFK-BP" (Edsberg ground, internal id SFKBP1109). It includes a 5-minute clip "SFKBP1109_s1200" (8,992 frames).
  - Six more training matches: Vasalund, Solheim, Djursholm, Spånga (night, striped kits), Solberga/Norrviken, Reymersholm.
  - All 6 training matches are now in our own storage (Cloudflare R2).

## 3. The current ball pipeline (as it runs in the app today)

```
video frames
  ├─ Finder 1: "click finder": YOLOv8 detector, 1 class "ball"
  │     start weights: roboflow/sports football-ball-detection (YOLOv8, AGPL-3.0)
  │     fine-tuned on Daniel's hand clicks (+ checked auto-labels, see §5)
  │     run at imgsz 1920 on the full frame + a 2x "far zoom" pass on the top 45% of the frame
  │     conf floor 0.05; duplicates within 12 px merged
  ├─ Finder 2: WASB (Tarashima et al., BMVC 2023, MIT code): 3-frame HRNet heatmap model for small fast balls
  │     frame halved to 960x540, cut into 2x2 overlapping tiles fed at 512x288 (ball ≈ 6 px, what WASB expects)
  │     soccer weights from the WASB repo (trained on ISSIA; weight licence unclear), fine-tuned on our clicks
  │     heatmap peaks above 0.05 kept (up to 6 per tile, merged within 6 px)
  ↓
Fusion (fuse_candidates): one guess list per frame. A guess both finders agree on (within 12 px)
  gets both scores + a bonus. Scores are squashed to 0–0.99.
  ↓
Picker v2 (pick_v2): Viterbi over frames. States per frame = top-12 guesses (conf ≥ 0.08) + a "no ball" state.
  emission cost = 1.5·(1−conf) + (on pitch: 1.0·(not within 4 m of a player) ; off pitch/in air: 0.6); "no ball" = 3.0
  transition cost in pixels after removing camera motion (H[i-1]·H[i]^-1): 0.02·px, +6 if the jump is >110 px, forbidden if >260 px
  static-clutter filter: a guess that stays at the same pitch spot for >60% of a 3 s window with nobody near is dropped
  (cones, signs, spare balls)
  ↓
Bridge: straight-line fill of gaps up to 1.0 s
  ↓
Export to app: per frame {pixel, pitch metres, state observed/bridged, confidence}.
  The ball layer is shown only when the checked accuracy is ≥ 60%.
```

An older picker, v1 (greedy track linking), and pick_global also exist. v2 is better.

## 4. How we measure the ball (our answer keys)

| Answer key | What it is | Size |
|---|---|---|
| **34 moments** | 34 frames spread over the 5-min SFK-BP clip, true ball position clicked by hand. "Right" = our pick within **30 px** of the truth. | 34 (small!) |
| **Ball exam** | 108 frames from the full SFK-BP match, held out from training. 81 have a visible ball, 27 have none. "Correct" = ball found within 30 px, or correctly no ball. | 108 |
| **Label checks** | Before using auto-labels for training, Daniel checks 20 random pictures per match. | 20/match |

**Weaknesses of this measurement:**
- Everything is on **one ground** (SFK-BP/Edsberg).
- 34 moments is too few: one moment = 3 percentage points.
- We don't yet grade "ball in play vs out" or "which player has the ball".

## 5. Training data we have

- **Daniel's clicks on SFK-BP:** 436 frames (328 train, 108 exam). This includes a special round of 150 "ball at a player's feet" frames (26 Sep).
- **Auto-labels from the click finder**, runs of 4+ consistent detections, checked by Daniel (20 pictures each):
  - Vasalund 19/20 ✔ (1,008 labels used)
  - Solheim 20/20 ✔
  - Spånga 19/20 ✔
  - Djursholm 20/20 ✔
  - Reymersholm 25/34 ✘, Solberga 7/11 ✘ (rejected)
  - Rule from Daniel: **spare balls lying next to the pitch must never count as the match ball.**
- **WASB-track auto-labels:** rejected. Only 3/28 were right in Daniel's check.
- **SoccerTrack v2** (CC BY 4.0, commercial use OK):
  - Panoramic fisheye video of real matches.
  - Its `mot/` files have **player boxes only**.
  - Ball positions are given **in pitch metres**, not pixels. The repo has a script (`scripts/calibration/project_tracking_to_image.py`) to project them into the image, but there are known y-flip issues.
  - Not used successfully yet. An earlier SoccerTrack attempt (round 8) wrecked the model; see §6.
- **Not usable commercially:** SoccerNet (research-only), SportsMOT (NC), TOTNet data (NC).

## 6. What we tried, and what happened (chronological)

| Date | Attempt | Result |
|---|---|---|
| ≤22 Sep | Picker v1 vs v2 on the 34 moments (WASB guesses) | v1 17–18/34, **v2 21–24/34** |
| 25–26 Sep | Ball exam, round 1: fine-tune the roboflow YOLOv8 ball model on Daniel's clicks | old model 48/108 → **66/108** (ball found 26 → 42 of 81) |
| 25 Sep | Analysis: every exam miss was a tiny far ball → added the 2x "far zoom" pass | helped a little on some rounds, not consistently |
| 26 Sep | Rounds 3–5: more clicks, incl. 150 at-feet frames | 50 → 52 → **68/108** (round 5) |
| 27 Sep | Round 6/7: + 4,000 auto-labels | 58/108, **worse** |
| 27 Sep | Round 8: + SoccerTrack v2 crops | **25/108, collapsed** (domain too different and/or bad projected labels) |
| 27 Sep | Picker test on the 5-min clip with cached guesses (older click finder) | WASB only 21/34 (ceiling 25) · clicks only 20/34 (ceiling 19) · **fused 25/34 (ceiling 27)** · fused + agreement bonus **26/34** |
| 27 Sep | Rule change: a model is promoted only if it beats the current best on the same exam | |
| 28 Sep | Round 9/10: clicks + 1,008 checked Vasalund labels | **70/108** (ball found 47/81) → promoted |
| 28 Sep | Clip run in the app with the new finder | **22/34** in the app (below the 25–26 offline result; cause not pinned down, see §8) |
| 28 Sep | Round 11: clicks + checked labels from 4 matches | 67/108 → **not promoted** |
| 28 Sep | Picker test with the *new* player tracking (better player positions) | unchanged, 21/34 (WASB guesses) |
| 28 Sep | "Nearness to a player" weight 2 or 3 in the picker | worse |
| 28 Sep | "Carry fill": ball disappears at player X's feet and reappears at X → place it at X's feet in between | no change |
| 28 Sep | **B1:** a "ball is with player P" state (every player's feet offered as a candidate at a fixed cost) | **no gain**: 18–21/34 depending on cost; best = unchanged |

**Important correction.** Several of our 28 Sep statements ("21/34, best possible 25, 8 of 13 misses have no guess on the ball") come from the offline tool `tools/picklab.py`. That tool only has **WASB** guesses saved locally, not the click finder's. With both finders fused (27 Sep test), the ceiling is 27/34, which means **7 of 34 moments where neither finder has any guess within 30 px**. The B1 test was also run on WASB-only guesses, so it should be re-run on fused guesses before we call it dead. The main conclusion still holds: *the finders miss the ball outright in ~20% of moments, and no picker can fix that.*

## 7. Current diagnosis

1. **Seeing, not choosing.**
   - Ball exam: the ball is among the click finder's guesses (down to conf 0.03) in only **52 of 81** ball frames (64%). It is the top guess in 47.
   - On the clip, the fused guesses contain the ball in 27/34 moments, and the picker gets 25–26 of those offline.
   - So the finder recall ceiling is ~64–80%, and the picker loses 1–3 points on top of that.
2. **The misses are the hard cases:**
   - far and tiny (4–6 px)
   - motion blur
   - ball against white lines/shoes
   - at a player's feet or partly hidden
   - in the air (projects off the pitch, so the pitch-metre logic mistrusts it)
   - A per-miss picture sheet is being made now to count each type (job B5).
3. **B1 showed:** in the 9 moments with no WASB guess, a detected player's feet are within 30 px of the ball in only 3. So most missing balls are **not simply "at someone's feet"**. They are loose, far, or in the air.
4. **More of our own labels helped until ~70/108, then stalled.** Adding other grounds' auto-labels (round 11) did not help the SFK-BP exam. We don't know whether that is label noise, too much data from one finder's own mistakes (self-confirmation), or the exam being single-ground.
5. **Licence risk:** the click finder starts from AGPL-3.0 YOLOv8 weights. For a commercial product we want to move to an Apache/MIT model (RF-DETR is Apache-2.0).

## 8. Open unknowns (I don't know yet)

- Why the in-app clip run scored **22/34** while the offline fused test scored **25–26/34**. Candidates:
  - the click-finder version differs (round 10 vs earlier)
  - the "far zoom" setting
  - the stride (the app runs the click finder on every 3rd frame for speed)
  - a caching difference
- How ball accuracy transfers to **other grounds** (no ball answer key off SFK-BP yet).
- Whether the finders "see" the missed balls weakly, just below the 0.05 cut-off. This needs raw heatmaps / low-threshold detections, which only run on our GPU setup (Modal), so it waits for approval to spend.
- How much of the pass/possession error is caused by each ball failure type.

## 9. Context: players (fixed on 28 Sep, offline)

This matters for any "ball near player" logic. Player tracking was replaced on 28 Sep:
- RF-DETR (Apache-2.0) COCO person detector instead of the AGPL football model. The old model labelled black shirts "referee", and tracking then deleted those players.
- Team colours are learned per match.
- The keeper is detected from the penalty area.
- Short gaps (<1 s) are filled, and duplicates removed.

Full 5-min clip results:
- dark players tracked for a median of **1.5 → 4.1 s**, light players **0.7 → 7.0 s**
- players missed per frame dropped from 2–4 to 0–1

Better players did **not** change the ball picker's score. This is not in the app yet.

## 10. What others do (research, 28 Sep)

- **Veo** (patent US 11,310,418):
  - per-pixel ball probability map
  - players detected and Kalman-tracked
  - **player posture predicts where the ball is relative to the player**
  - maps combined across players
  - a **particle filter** (≥100 particles, Newtonian motion) picks the ball
  - Veo's blog admits "anything white would be recognized as a ball".
- **Spiideo, Pixellot:** installed, calibrated fixed cameras, so an easier problem.
- **SkillCorner:** extrapolates off-screen objects and flags `is_detected`.
- **Academic:**
  - WASB (small ball, heatmaps, 3 frames)
  - TrackNetV3 (MIT) with an "InpaintNet" to fill occluded trajectories
  - TOTNet (MIT code, NC data), temporal occlusion
  - arXiv 2506.07981: a 4-state possession model (in play / with player / after possession / out), 66% within 1 m on CPU
  - PathCRF (MPL-2.0): infers ball path from player movement only
- **Data scale:** WASB soccer used ~12k labelled frames. We have ~436 hand-labelled + a few thousand checked auto-labels.

## 11. Constraints

- **Budget:** no GPU spend (Modal) without the founder's explicit "go", even under $1. Prove things offline or on free GitHub runners first.
- **No manual work for the founder:** he will not click more labels at scale, will not provide Veo's stats, and will not type set-piece times. Label checks of ~20 pictures are acceptable.
- **Licences:** everything shipped must be commercial-OK (MIT/Apache/BSD/CC BY). No SoccerNet data, no AGPL long-term.
- **Accuracy must hold across all our grounds**, not one clip.
- A new model replaces the old only if it scores better on the same exam.

## 12. The plan (ranked), with status

| # | Step | Cost | Status |
|---|---|---|---|
| B1 | "Ball with player" state in the picker | free | done, no gain on WASB guesses; **re-run on fused guesses** needed |
| B5 | Picture sheet of all misses, sort by cause (far/blur/feet/air/crowd/lines) | free | running now |
| B6 | Ball answer key on a second ground from existing clicks | free | queued |
| B4 | Grow the answer key 34 → 300+ moments without clicking (frames where both finders and the picker agree strongly, checked by eye by Claude) | free | queued |
| B2 | Look below the cut-off: do the finders weakly see the missed balls? Top-K + 2×2 tiling + flip test-time augmentation for far balls | ~cents on GPU | needs approval |
| B3 | SoccerTrack v2 ball labels projected to pixels (ground balls only, fix the y-flip), checked on a picture sheet before any training | free (CPU) | queued |
| B7 | Ball finder on **RF-DETR** (Apache) trained on clicks + checked auto-labels + (B3); dry-run on CPU first | < $10 GPU | queued; training needs approval |
| — | Active learning: label frames where the two finders disagree; self-training only on long confident stretches | small | idea |
| — | Temporal occlusion model (TrackNetV3 InpaintNet / TOTNet-style) | later | after the finder improves |
| — | Particle filter + player-posture cue (Veo-style) instead of Viterbi | free | idea; needs pose/orientation of players |

## 13. Questions we'd like a second opinion on

1. With a **moving follow-cam** at 1080p and 4–6 px far balls, is a single-frame detector (YOLO/RF-DETR) at high resolution with tiling, or a multi-frame heatmap model (WASB/TrackNet-style) the better core finder? Or both, fused as now?
2. How would you get **5–10k good ball labels** with almost no human clicking, and without self-training on your own errors? Would you trust SoccerTrack v2 (fisheye panorama, projected metres → pixels) for a follow-cam domain after round 8 collapsed?
3. Is Viterbi over candidates the right picker, or should we move to a particle filter with ball physics (ground vs air) and player-possession states as in Veo's patent?
4. How should an **airborne ball** be handled, when projecting it to the pitch plane gives nonsense positions?
5. Which **evaluation** would you trust for "90% ball accuracy"? Per frame within 30 px? Per possession segment? Per event (pass/turnover correct)?
6. For **stats** (passes, possession), would you rather infer events from **player movement** (e.g. PathCRF-style) and use the ball only as a check, given the ball will never be visible 100% of the time on this footage?
7. Any commercially licensed (MIT/Apache) **pretrained soccer ball weights** or datasets we have missed?

---
*Source files in the repo:*
- `ipanema/ball.py` (pickers, fusion, bridge)
- `ipanema/wasb.py`, `ipanema/wasb_train.py`
- `ipanema/ballclicks.py` (click finder)
- `tools/picklab.py`, `tools/posslab.py` (offline tests)
- `results/ball/*/summary.json` (exam rounds)
- `results/ball/pick_test.json` (fused test)
- `results/research/ball_2026-09-28.md` (research)
