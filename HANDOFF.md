# Ipanema — handoff (written 23 Sep 2026, end of the long "follow-cam calibration" chat)

Read this first in a new chat. The previous chat is in the IPANEMA project and can be searched word for word
(search e.g. "follow-cam calibration", "clicks", "review_session", "propagate").

## Goal and hard rules (from Daniel)
- Goal: SFK–BP (match id `SFKBP1109`, Veo follow-cam, 102.7 min, 1920x1080, 29.97 fps) analysed to consumer standard:
  ~90% accurate stats, overlays that make sense, ball tracked properly.
- **The consumer product uses ONLY the Veo follow-cam footage.** The screen-recorded panorama is internal reference/training only.
- **No money spent on Modal for untested ideas.** Develop and prove offline (sandbox, GitHub's free runner, Colab free GPU),
  show pictures, only then a cheap gated run with Daniel's explicit go. Judge by pictures, not by scores.
- Be concise, honest, no guessing presented as fact.

## Where the problem stands
- Positions depend on calibrating each follow-cam frame. The follow-cam = the same fixed Veo camera, turned and zoomed:
  per frame only pan, tilt, roll, zoom vary.
- **Camera base (settled, from Daniel's clicks):** `calibration/panorama/SFKBP1109_base.json` — on the halfway line,
  ~3.6 m behind the near touchline, ~4.8 m up, small base tilt. Solved jointly from 56 clicked frames (319 points).
  Hidden-point check (predict a hidden click from the rest): midfield 5.0 px, right end 6.7 px, left end 7.8 px median
  (1280-wide frames). Penalty-spot clicks are unreliable and excluded. Pitch 106 x 64 m, standard markings (verified).
- Per-frame poses of the 56 clicked frames: `calibration/panorama/SFKBP1109_clicks_solution.json` (1280x720 poses).
- Frame-to-frame tracking (`ipanema/ptz.py` `track_sequence`, optical flow + line refine `ipanema/fccam.py refine`, no
  snapping when no lines) is reliable over seconds (tested 120/120 on real windows; propagation test 30/30).
- **Unsolved:** automatic per-frame calibration from the picture alone (single-frame line fit + judge fail on corner/near views).
  Current approach: train a pitch-point detector on Daniel's labels.

## Pitch-point detector (in progress)
- v1 (YOLOv8s-pose, 47 train / 9 held-back clicked frames, Colab T4): held-back grade **0/9 frames with 2+ correct points,
  median 19.3 px, 29% within 15 px, 0 confident mistakes** — too little data. Weights: Drive `ipanema_models/pitch_kp_v1`.
- **Step 1 DONE (23 Sep ~20:25):** `kptrain.propagate` carried the 47 training frames' calibrations ±3 s →
  ~1,400 proposed frames: Drive `ipanema_labels/SFKBP1109_proposed.zip` (340 MB) + `SFKBP1109_proposed.json`.
- **Step 2 NOW:** Daniel reviews proposals YES/NO with `label.review_session` (fixed to load one picture at a time).
  Output: Drive `ipanema_labels/SFKBP1109_review.json`. Target ~300 answers; if the first 20–30 are mostly NO, stop and
  investigate propagation (which seed frames fail).
- **Step 3 NEXT:** `kptrain.merge_reviewed` + `build_dataset` (only clicked frames are ever held back) → train v2 (cell in the
  chat, same settings, project `ipanema_models`, name `pitch_kp_v2`) → `kptrain.evaluate` on the same 9 held-back frames.
  If still weak: switch to per-point detection (heatmap / one small box per point) instead of one 31-keypoint pose object.
- Later: click session 3 / second-half frames for variety (all clicks so far are 0–43 min).

## Colab cells (Drive paths)
- Labels folder: `/content/drive/MyDrive/ipanema_labels` (frames zips s1, s2; points s1, s2; proposed; review).
- Full match in Drive: `p15u-2026-vs-bp-1133-2026-09-12.mp4` (3.6 GB; found by glob under MyDrive).
- Every cell starts by re-cloning `https://github.com/danielzh1nt0/ipanema-analysis.git` into `/content/ipanema-analysis`.
- Claude can read small files from Drive via the Google Drive connector (JSON yes; big zips / model files no).

## Other state
- Ball on follow-cam: ~65% correct on held-out (picker v2 24/34 on the bench). Ball on panorama unusable.
- Veo shots/goals imported correctly (25 shots, 3 goals, 2-1). Periods 0:00–51:00, 1:02:13–1:39:00.
- Panorama clip `p15u-vs-bp-2026-09-22-2000` (screen recording, cropped to Veo's player): internal only.
- Automation: tests (42) + model evals with floors on every push, release log, per-stat QA verdicts, speed watchdog,
  one-command match, live progress into the app, Veo highlight-name parser.

## Lessons (do not repeat)
- float32 projections break optimisers (use double precision); reject points behind the camera; principal point = frame centre.
- The frame judge must run at the size it was validated at (size-aware now); never trust it without looking at pictures.
- Never embed many images in one Colab page (340 MB page was truncated) — fetch one at a time.
- Old calibrations (120x70 model) poisoned the camera base; the panorama fit's position is not reliable for the follow-cam.
- Budget/time estimates on Modal were repeatedly wrong: measure first, save progress incrementally, one core per CPU piece.


## Update 24 Sep 2026 (end of the long chat)
- **v2 pitch-point model** (658 frames = 47 clicked moments + 611 approved propagated neighbours): WORSE — held-back 0/9 placed,
  median 77 px (best.pt) / 1 of 9, 187 px (last.pt), 14-22 confident mistakes. Cause: memorised 47 moments (propagated frames
  are near-copies); held-back frames are other moments. Lesson: **variety of moments, not volume.**
- **Random-moment proposals** (`kptrain.propose_random`, single-frame line fit + judge, avoiding labelled times): 420 proposals
  (Drive `ipanema_labels/SFKBP1109_random.zip/.json`); Daniel reviewed all: ~180 YES / ~240 NO
  (`SFKBP1109_random_review.json`), both halves covered. So the automatic fit+judge is wrong ~60% of the time even when
  "confident" — never trust it without review. The NO frames are useful hard negatives.
- Verified distinct moments now ~230 (47 clicked-train + 9 held-back + ~180 random YES) + ~580 propagated YES neighbours.

## Next steps (agreed)
1. **Line-based model instead of point regression**: train segmentation of painted-line classes (touchlines, goal lines,
   halfway, box/goal-area lines, circle) from verified poses (render the 106x64 model lines through each verified pose as
   masks). Then per frame: fit pan/tilt/roll/zoom with the KNOWN base to the predicted line masks (fccam Scorer with the
   predicted mask instead of line_mask), temporal tracking re-anchored every few seconds. Build + test offline first
   (render masks, check overlays), then one Colab training cell (free T4; torch must be 2.11.0+cu128 — Colab's cu130 build
   crashes on T4: `pip install --force-reinstall "torch==2.11.0" torchvision --index-url .../cu128`, then restart via
   `os.kill(os.getpid(), 9)`).
2. **Corner-only click session** (~60 corner/near-side frames, both ends, both halves) — random proposals barely cover them.
3. Grade every round on the same 9 held-back clicked frames (add held-back corner frames too).
4. Ideas discussed: per-match camera base is solved automatically from detections (model learns appearance, not position);
   synthetic views from other camera positions; AI-agent YES/NO review (test agreement on Daniel's 1,157 answers first,
   API cost a few $); Upwork/CVAT labelling of 4 more matches (~250-300 varied frames each, auto-QA via camera-consistency
   check; footage shows minors -> club permission + GDPR data agreement first).
