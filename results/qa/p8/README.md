# P8 (29 Sep): white players read as the green team at night - per-player team classifier

Tool: `tools/p8lab.py` (free, local). Code: `ipanema/kits.py` (`body_hist`, `fit_player_cls`), on by default (`IPANEMA_KIT_CLS=1`).

## First: the free-runner check from P7 (results/qa/p7_track_on vs p7_track_off, Reymersholm 20 s from 1500 s)
- Same numbers with and without the off-pitch drop: dark 6 / light 4 per frame, 51 vs 52 track ids. In that piece the
  off-pitch people were already left out by the kit step, so the drop changed nothing there. By eye the teams look right in it.

## What was wrong
- The kit reading uses the MEDIAN colour of the middle of the shirt. For a small, blurred, floodlit white player that median
  is a mix of shirt and bright grass, so it lands nearer the green kit. The crops are clearly white by eye.

## The fix: a small per-match classifier, no labels needed
- Each person gets a colour histogram of the upper body (5x5x5 bins, lightness scaled to the grass). A blurred white shirt
  still has many bright pixels, which the histogram keeps.
- Trained per match from the sample frames: group the on-pitch people by histogram; the biggest group the colour model
  calls team A and the biggest it calls team B are the training examples; 15 such runs vote.
- Safety check: both teams have about as many players on the pitch. A run is only used if its team split is nearer 50/50
  than the colour model's. Otherwise the colour reading stays (this is what keeps Spånga and SFK unchanged).
- Referees, keepers and staff ('other') still come from the colour model.

## Results (graded by eye)
| ground | colour only | with classifier |
|---|---|---|
| Reymersholm night, 252 players | 195 right, 51/115 whites read green | **239 right, 7/115 whites read green**, 4/137 greens read white |
| Spånga night, 279 players (new key, `results/qa/kitprobe/spanga_labels_p8.json`) | 181 right | 181 (classifier switches itself off) |
| SFK day (no key) | - | 0 people change team (switches itself off) |
- Without the 50/50 check the classifier hurt Spånga (162-177/279), where the striped kit's histogram overlaps the black kit's.
- Pictures: `*_moved_AtoB.jpg` = people the classifier moved (Reymersholm: almost all white players, a few spectators),
  `*_kits_new.jpg` = the learned teams.
- Dry run of the tracking (stand-in detector = the saved boxes, 24 Reymersholm frames): dark/light per frame 8/2 -> 6/5.

## Also found
- Spånga: 79 of 279 players are read as 'neither team' (mostly striped players). That is P1's problem, not changed here.
