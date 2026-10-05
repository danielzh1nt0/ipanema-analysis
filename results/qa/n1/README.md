# N1 - nightly read, 5 Oct 2026 (worker, local, $0)

## What the 5 Oct card says vs 4 Oct
Ball 22/34, passes 15% off, possession 8 pts, sequences 40%/44%: all unchanged.
Tests 88 -> 100 (new tests, all pass). One thing got worse:

**Solberga (night, no calibration): light players seen per frame 7 -> 6.**

## Why
The nightly re-tracks 20 s of Solberga on the free runner. Logs of the two runs:
- 4 Oct: `keeper (right) from a non-team kit: track 1, 267 frames` -> kept as a keeper.
- 5 Oct: `no calibration: keeper-by-goalmouth rules off` -> track 1 removed with the referee/staff rows (908 -> 1175 rows removed).

Track 1 is **white #5, a real outfield player, near the camera on the right** (pictures:
`solberga_before_after_0000.jpg`, `solberga_before_after_0256.jpg`; top = 4 Oct, bottom = 5 Oct; blue ring on him only on top).
He was never a keeper: the screen-edge keeper rule (switched off in P2e for grounds without calibration) had been
rescuing him by luck. The real fault is older: under the floodlights the kit model reads the brightest near whites as a
"non-team kit" (referee / staff). A near white on the left has no ring in either run (player or staff - can't tell at this size).

Other key-frame changes are id renumbering (40 -> 9/55) and one far player flipping team at frame 256 (can't tell by eye).
App untouched (the app's grounds have calibration; this is the tracktest path only).

## Built
`tools/nightly_scorecard.py` now adds a row `| Worse than <previous day> | ... |` to every card: ball, pass/possession
error, sequences, failing tests, players seen per frame (drop of 1+) and median tracked time (drop of 20%+).
Dry run on the repo's 4/5 Oct files: output identical to the old script plus
`| Worse than 2026-10-04 | solberga-vs-p09-norrviken-2026: light players seen per frame 7 -> 6 |`; history.tsv unchanged.
Tests: `tests/test_n1_nightly_compare.py` (6). Full suite here: 280 passed; 5 fail only because this machine has no
torch / modal (ball scorer, tiles, v3 frames, watchdog, panorama texture) - none touch these files.

## Next
New queue item N1a: near floodlit whites read as non-team kit on Solberga (and the night grounds) - fix in the
non-team-kit removal, grade on the Solberga key frames; P2e stays.
