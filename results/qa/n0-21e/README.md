# N0-21e (21 Oct 20:20 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card 21 Oct:** worse than 20 Oct = nothing. Coverage run of the app code: the RF-DETR ball finder's own code (`ipanema/ballrf.py`, every app run's ball guesses) was 18% tested.

## Findings
1. **Guesses at the tile edges.** The finder looks at 8 overlapping 640 px tiles. In the app finder's guesses for the whole SFK-BP clip (8,992 frames) there are 2-3x more guesses just inside a tile's inner side than a few px away (y 442-446, y 628-634, x 634-636, x 1282-1286): the finder fires on things the tile cuts in half.
2. **Wrong pixels for a video that is not 1920x1080 (latent).** Frames are resized to 1920x1080 before the finder, but the guesses were never scaled back, while calibration, players and WASB use the video's own pixels. All app videos so far are 1920x1080, so nothing in the app changed.

## Changes
- Fix 2 is on: guesses go back to the video's own pixels (no change for 1920x1080).
- Fix 1 is an option, **off** (`IPANEMA_BALL_EDGE=1` = on): a guess whose box touches a tile's inner side is dropped. Every such spot lies at least 200 px inside a neighbouring tile, which sees a real ball there whole.
- `tests/test_ballrf_edge.py` (10): tile layout, inner vs picture edge, tile offsets + merging over a batch, the 20-guess cap, the rule off/on (cut junk dropped, a ball cut by one tile kept from the other), scaling back in both readers, tool dry run.

## Check on the free runner (`tools/n0_21e_balledge.py`, real app finder, 33 min CPU)
Key frames: SFK-BP 319 (34 checked + B4 balls), AIK 39.

| | SFK-BP | AIK |
|---|---|---|
| guesses per frame, old -> new | 10.8 -> 10.5 | 9.3 -> 9.0 |
| key ball among the guesses | 319 -> 319 | 39 -> 39 |
| top guess on the ball | 308 -> 308 | 28 -> 28 |
| top 3 on the ball | 316 -> 316 | 39 -> 39 |

By eye on all 127 dropped guesses (`dropped_00.jpg`, `dropped_01.jpg`; red = dropped, green = key ball, cyan ticks = tile sides): almost all are junk (heads, shoes, knees, plain grass, lines). One was the key ball itself (SFK 3339, conf 0.74). The other tile's whole view of that ball stayed top guess. Maybe 1-2 other ball-like spots (e.g. SFK 7942, far right).

**Verdict:** it removes ~3% of guesses, nearly all junk, and loses no ball. But on single frames it changes no top guess, so a gain is not proven, and it **stays off**. A fair test needs the whole clip re-detected so it can go through the picker and the 34/B4/who/pass keys. On the free runner that would take ~14 h of CPU (358 frames took 33 min), so it needs a free Kaggle GPU job (~30 min).

Note: with torch installed locally, `tests/test_tracking_batch.py` fails in the full suite because another test leaves a stand-in `ultralytics` module in place. It passes on its own and on the nightly runner (no torch). Nothing to do with this change. Suite otherwise 536 pass.
