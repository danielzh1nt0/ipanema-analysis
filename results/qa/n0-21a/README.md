# N0-21a (21 Oct 00:20 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card:** 20 Oct says nothing worse than 19 Oct (21 Oct card not due before 03:00). Full local suite: 515 pass, 2 skipped.

## Finding: player boxes cut at a tile's edge, feet put too high
The app's player detector (RF-DETR, `tracking.RFDetrPerson.detect_batch`) looks at 3 tiles: the whole frame, and the far band (20-65% of the picture height) in two halves. A player standing just below the band comes out of the band tile with a box that ends at the band's edge, at his knees. The whole-frame tile usually finds the same player uncut, but the merge (NMS) keeps whichever box is more confident, and the zoomed band tile is often more confident.

- In 37 saved RF-DETR tracking runs (results/qa/**/rows_all.json.gz, 536,765 tracked feet), **15,399 feet sit exactly on y = 701 of 1080** (the band edge). The 50 px below it are nearly empty: ~37 per px there vs ~450 per px just above.
- By eye on 36 saved boxes ending on that line, most stop at the knees or shins.
- The same thing happens, less often, at the band halves' inner side edges (x = 864 / 1056): half a player.

## Fix (on by default, `IPANEMA_TILE_CUT=0` = old)
`tracking.drop_cut`: a box that touches an edge of its tile (within 2 px), where that edge is not the edge of the picture, is dropped when a box from another tile that is not cut itself covers at least 60% of it. A cut box with no such partner is kept, because it is still a person.

## Check on the free runner (tools/n0_21a_tilecut.py, the real RF-DETR medium on 164 saved frames: 18 pieces from 6 grounds plus SFK-BP and AIK)
- Boxes 3,327 -> 3,326. Boxes ending on the band edge 77 -> 14. 79 boxes changed (65 bottom cuts, 13 side cuts), 1 dropped.
- Bottom-cut feet move down a median of 36 px (up to 79 px).
- By eye (`sheet.jpg`: red = old, green = new; `zoom_doubtful.jpg`): about 71 clearly better (now the whole player, feet included), 7 about the same (two overlapping players, a crouching player, a few px change), 1 can't tell (dark night spot), **none worse**. The one dropped box was a thin piece between the referee and a player.
- Note: the AIK frames in results/qa/players already have old boxes drawn on them. They make up only 3 of the 79 changes.

Left over: 14 cut boxes per 164 frames have no whole-frame partner, so their feet are still on the edge. A real fix for those needs the band to reach lower, or the box extended. Not done.

Reaches the app at the next tracking run (players GPU run, Modal - Daniel). Tests: `tests/test_tile_cut.py` (9: band-bottom cut dropped with a partner, kept alone, uncut and frame-edge boxes untouched, inner side edge, partner must cover the box, switch off, detect_batch end to end with a stand-in detector (old keeps the cut box, new the whole player), tile offsets over a 2-frame batch, tool dry run).
