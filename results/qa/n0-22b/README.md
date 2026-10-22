# N0-22b (22 Oct 05:00 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card:** no 22 Oct card by 04:20 -> backup nightly pushed; card arrived: worse than 21 Oct = nothing.
**Queue before this:** M2 (main session), PC2 (waits for M2), BALL-EDGE (needs a Kaggle GPU; this worker runs only local + the free runner) - skipped, left open.

## Finding
The follow-cam calibration (`mosaic.calibrate_via_mosaic`, used by both app test clips) samples every 10th frame and gives
EVERY frame a homography: where a sample can't be registered, the frames in between get a straight blend across the gap,
and frames before the first / after the last registered sample get that sample copied. The app then counted every frame
as calibrated - log `calibration: from panorama, 8992/8992 frames registered`, `calibration_coverage` 1.0, frozen 0 -
so the full-match QA check "calibration > 95%" (which decides the 'positions' verdict) could never fail on this route.

## Fix (on by default, `IPANEMA_MOSAIC_COVER=0` = old count)
- `mosaic.calibrate_via_mosaic(..., anchors_out={})` returns which samples were registered (direct / chained).
- `mosaic.measured_mask`: a frame is measured when its two neighbouring registered samples are at most ~1 s apart
  (one missed sample is fine), or it is within one sample of the first / last; everything else is a guess.
- `run.prepare`: coverage / frozen now count measured frames only; the anchors are saved next to the calibration cache
  (`calibration_pano_<ver>_anchors.json`). An old cache without that file keeps the old count.
- No homography changes - only the number and the log line.

## Real clips (free runner, `tools/n0_22b_mosaic_gaps.py`)
| clip | samples registered | direct / chained | guessed frames |
|---|---|---|---|
| SFK-BP (SFKBP1109_s1200) | 899 / 900 | 267 / 632 | **0** |
| AIK (p15u-vs-aik_s2520) | 900 / 900 | 530 / 370 | **0** |

So on the two app clips nothing changes (coverage stays 100%); the fix only matters on a clip where registration breaks
(e.g. the camera blinded / pointing off the panorama - the test makes one: 109 of 300 frames guessed, old count said 300/300).

By eye on the picture sheets (`*_sheet.jpg`, model in red): SFK-BP 4/4 fit the painted lines (halfway line slightly
tilted near the camera); AIK 4/4 look right (centre circle, far box) - but the automatic line judge calls 477/900 SFK-BP
and 743/900 AIK frames 'bad'. Not trusted here: it is run with the app's 120 x 70 model (lines past a smaller real pitch
fall off the grass) and AIK has a big shadow over the near half. Noted, not acted on.

## Tests
`tests/test_mosaic_cover.py` (5): mask rules; the real registration on a stand-in clip with a 4 s blinded stretch
reports the gap; prepare counts only measured frames (and again from the cache); old switch + old cache; tool dry run.
Suite here: **545 passed**, 0 failed.
