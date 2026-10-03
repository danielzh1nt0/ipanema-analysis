# S3: pasted balls (3 Oct 2026, local, free, no training)

**Result:** built. 414 pasted balls on 69 of our frames (SFK-BP, Reymersholm, Spånga night), 4-8 px, far side; 117 sit beside a player's feet and 35 on or near a line. By eye they are **not yet realistic enough**: in a blind test I could still pick out pasted balls 43-44 times out of 48.

## How it works
- `ipanema/pasteball.py`, `tools/s3_paste.py`, `tests/test_pasteball.py` (6 tests).
- **Source balls:** 325 checked balls (SFK-BP training clicks: 260, the test clips are never used; K1 checked balls from 4 more grounds: 65). 84 of them are clean enough to cut out: round, on plain grass, with a bright panel. Most rejects have legs or lines next to the ball.
- **Size:** taken from the players in the same frame. A ball is about 0.12 of a player's box height at the same row, so it is right for that spot's perspective.
- **Spots:** on the pitch's grass, never on top of a person, weighted towards the far side. Half on open grass, a quarter beside a player's feet, a quarter near a line.
- **Look:** the ball's colour and brightness are matched to the grass at that spot (floodlights are yellow). It also gets a light blur, motion smear on 30% (a moving ball), a soft ground shadow on half, half-resolution colour like the video codec, and a JPEG round trip of the small window around it.
- **Rebuild:** `labels.json` lists every pasted ball (frame, x, y, size, box, source crop, kind). Running `python tools/s3_paste.py 6 0` rebuilds exactly the same pictures.

## Grade (by eye, blind)
`sheet_blind.jpg` mixes 24 small real balls with 24 pasted ones at the same sizes, shuffled. I guessed each tile first and only then opened `blind_key.json`.

| round | change | right out of 48 |
|---|---|---|
| 1 | heavy blur + smear | 42 (all 6 floodlit Spånga tiles were pasted, an easy tell) |
| 2 | light blur, sharpening | 42 (dark shadow blobs had been cut as "balls") |
| 3 | drop dark blobs, light colour matching | 28, but only because I wrongly assumed the orange (Spånga) tiles were real. Without them: 28/37 |
| 4 | Spånga left out of the blind test | 43 |
| 5 | shadow + codec colour + JPEG | 44 |
| 6 (kept) | + spots at feet and near lines | 43 |

**What still gives them away:**
- Pasted balls look like soft lime blobs with a shadow, or like crisp "stickers".
- Real small balls usually have something next to them (legs, a line).
- Reymersholm's speckled grass looks different from every ground where real balls come from.

**Caveat:** this test is weaker than it looks. The 48 smallest real balls were the same in every round (only 17 real checked balls are 9 px or smaller), so by the end I partly recognised them.

## Before any training (S3b)
- The base frames must be frames whose real ball is already labelled (the 4,511 labelled frames on R2/Kaggle). The kitprobe frames used here have the real ball unlabelled (visible in `frame_SFKBP1109_s1200.jpg`), and that would teach the finder to ignore real balls.
- Whether pasted balls help can only be shown by a free Kaggle training run, graded on the exam (B7: 84/108) and the clip (34 moments). The eye check says they do not pass as real; a detector may still learn from them.

Files: `sheet_real_vs_pasted.jpg` (top: real, bottom: pasted), `sheet_blind.jpg` + `blind_key.json`, `frame_*.jpg` (pasted balls circled), `zoom_*.jpg` (2x, a pasted ball at the centre), `summary.json`.
