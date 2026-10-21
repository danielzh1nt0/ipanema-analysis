# N0-21c (21 Oct, worker, local $0, nothing on Modal)

**Card:** results/nightly/2026-10-21.md says nothing worse than 20 Oct (tests 147/147, ball 22/34, all player checks ran).

**What was looked at:** a coverage run of the whole suite (516 pass) to find app code with no test. `run.prepare` - the
app's per-clip step that picks the calibration, where the players come from, which ball guesses are used and which
cache is read or written - had none (run.py 41% covered, prepare 0%).

**New test:** `tests/test_prepare_routing.py` (7). Every model / video step is a stub; the test checks the path taken:

| case | expected | result |
|---|---|---|
| app default: RF-DETR tracks + RF-DETR ball guesses in the cache | RF players re-positioned, RF + WASB guesses fused, no new detection | right |
| no RF tracks | old detector once, then its cache on the 2nd run; mosaic cached too | right |
| click model present, no RF guesses | click model + WASB; RF guesses win once they exist | right |
| WASB-only | the side-by-side 'WASB + YOLO' check is built | right |
| app default + an old YOLO cache | no 'WASB + YOLO' check | **was wrong** (fixed) |
| fixed camera (calibration/fixed/<match>.json) | no mosaic, own track cache, metres from its own points | **was wrong** (fixed) |

**Fault 1 (fixed, a log line only):** the 'alternative: WASB + YOLO' ball check was built from whatever guesses the app
used. Since 29 Sep those are RF-DETR + WASB fused, so it rescaled the fused scores as if they were WASB peaks and logged
the result under the wrong name - the live SFK-BP clip run (18 Oct, results/live/SFKBP1109_s1200.log) shows
"alternative: WASB + YOLO candidates: 14/34". Nothing else reads it: no stat, export or app number changes. Now only
built when the main guesses are WASB-only (IPANEMA_ALT_WASB_ONLY=0 = old). Saves one extra whole-clip ball pick per run.

**Fault 2 (fixed, latent - no fixed-camera match exists yet):** a fixed-camera match (D1-style tactical clip) still ran
the mosaic registration first (a full pass over the video, then ignored), and its tracks were cached as
tracks_pano_<v>.pkl / tracks_kp.pkl - the same names a follow-cam run of that clip uses - so tracks made with another
calibration would have been loaded with their old metres. Now: mosaic skipped, own cache tracks_fixed.pkl, earlier
detections reused with metres from the hand-read points, and cached fixed tracks re-positioned every run (cheap, so an
edited point file never leaves stale metres). IPANEMA_FIXED_CACHE=0 = old names.

Both tests fail on the old run.py (4 of 7) and pass on the new one. Full suite: 523 pass, 4 skipped.
No pictures: nothing drawn or detected changes. Reaches the app at the next run (Modal, Daniel).
