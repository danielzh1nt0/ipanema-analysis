# B3b - SoccerTrack v2 ball labels checked by our ball finder (3 Oct, free runner, $0, no Modal)

**Result: 254 of the 717 B3 labels kept, and by eye 242 of them are clearly a ball (12 can't tell at sheet size,
none clearly wrong).** B3 alone gave about 1 good label in 10 (~70); this gives ~240 clean ball pictures from
9 SoccerTrack matches, ball 3-10 px (median 5 px), 169 of them at a player's feet.

How (tools/b3b_finder.py, ipanema/b3b.py, tests/test_b3b_finder.py):
- For each B3 label: the video frame, a 640-px window around the projection at 1x and at 2x zoom, the app's
  RF-DETR ball finder (B7) on both.
- Keep when the best guess within 60 px of the projection has confidence >= 0.5 and no second guess there is
  >= 0.6 x as strong. The label moves to the finder's ball (median 25 px from the projection, 90% within 52 px).
- 18.8 min on the free runner (CPU). Dry runs first: stand-in finder (test) and the real finder on 6 SFK-BP frames.

Numbers (summary.json): kept 254, no guess near the projection 324, weak guess 134, two guesses 5.
Seen at both zooms 217, only at 2x 36, only at 1x 1.

By eye (kept_<match>.jpg, best first; green ring = finder, yellow tick = projection):

| match | kept | ball | can't tell |
|---|---|---|---|
| 117093 | 31 | 30 | 1 |
| 118575 | 25 | 22 | 3 |
| 118576 | 29 | 26 | 3 |
| 118577 | 18 | 17 | 1 |
| 118578 | 28 | 28 | 0 |
| 128057 | 31 | 30 | 1 |
| 128058 | 38 | 38 | 0 (2 are spare balls by the goal) |
| 132831 | 13 | 12 | 1 |
| 132877 | 41 | 39 | 2 |
| **all** | **254** | **242** | **12** |

The "can't tell" ones are nearly all 2x-only hits in crowds, by the stands or on a line. For training, the safe
set is the 217 seen at both zooms.

What is dropped (drop.jpg, random 15%): mostly no ball anywhere in the window - the tracked ball was further off
than 60 px or the moment is out of sync; a few real balls at feet with a weak guess (0.18-0.39).

Caveat: these are balls our finder ALREADY sees with confidence, so they lean easy. They add new grounds and
camera (fisheye panorama, 4K) but not the hard balls (half hidden at feet, in the air) the finder misses on our
footage. Worth adding only to a retrain that also has hard examples; on their own they are unlikely to move the
exam. No training run done.

Files: labels.json (254 kept, B3 fields + finder x, y, conf, scales), hits.json (all 717 with the guesses),
crops64.npz (X = 64x64 around each kept ball), kept_<m>.jpg, drop.jpg, summary.json, log.txt.
Data licence CC BY 4.0 (credit SoccerTrack v2).
