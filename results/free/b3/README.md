# B3 - SoccerTrack v2 ball labels in pixels (3 Oct, free runner, $0)

**Result: built, but the labels are NOT ready for training.** The dataset's ball position is too rough to cut
ball pictures from without a second check.

What was done (tools/b3_soccertrack.py, ipanema/soccertrack.py, tests/test_b3_soccertrack.py):
- The dataset's own chain: pitch keypoints -> fisheye camera -> tracking XML positions projected to the 4K picture,
  with its known fixes (132831 swapped keypoints, y flips on 132831/132877, 132877 ball not flipped, 0.5/0.5 placeholder).
  Our camera fit gives the same error as the dataset's notes on all 10 matches (8.6-30.6 px), so the calibration is right.
- Ground balls only: ball within 2 m of a player, or rolling slowly and steadily. 40 moments per half, 10 s apart.
- **Sync problem found** (sync.json, sync_*.jpg): in 13 of 18 halves the tracking runs 15-30 frames (~1 s) ahead of
  the video. Found per half from the players: projected feet vs a median background, 72-89% of players line up after
  the shift (42-64% before). The night match 117092 never lines up (lights, wet pitch) -> skipped.
- The dataset's `ball/*.npz` files are pitch metres on a 1.05 m x 0.68 m grid with a status code, not pixel labels.
- Each label is snapped to a ball-sized blob within 2.5 ball widths of the projection.

How good (by eye on sheet_117093 + sheet_128057, 160 tiles; yellow tick = projection, cyan ring = snap):
- The ball is usually inside the 80-px window, but typically 10-40 px (4K) from the tick: the tracked ball is
  ~0.5-2 m off (coarse grid + lag).
- The snap fires on 215 of 717 labels and is right on roughly 1 in 3 of those; the rest sit on shoes, socks and lines.
  So only about 1 label in 10 is a checked-good ball picture.

Files: labels.json (717 labels: match, half, video frame, raw + snapped pixel position, ball size, kind, sync shift),
crops64.npz (X = 64x64 around each label, N = ball-free crop from the same frame), sheet_<m>.jpg, over_<m>.jpg,
summary.json, sync.json, pass1_summary.json (first pass without the sync fix: balls mostly on empty grass).

Next, if wanted (low priority, see BACKLOG B3b): let our RF-DETR ball finder look in a 60-px window around each
projection on Kaggle (free GPU) and keep only confident hits, then an eye check of the sheet before any training.
Data licence CC BY 4.0 (credit SoccerTrack v2), code MIT.
