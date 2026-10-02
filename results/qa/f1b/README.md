# F1b — a kit model per 5-min piece (2 Oct, free runner, CPU, $0)

Question: the full AIK run uses ONE kit model for the whole match and calls ~2x more real players "neither" than the
clip's own model (results/ball/f1/README.md). Does a kit model per 5-min piece, its teams named like the match model, fix it?

Code: `kits.piece_model` (piece model from 36 frames of the piece; every person it puts in a team is also given a team by
the match model, majority decides whether the piece's A/B are swapped; falls back to the match model when the piece has
too few people or no clear majority). Not wired into the full run (no Modal); it would replace the copied match model in `modal_app.rf_piece`.
Test: `tests/test_f1b_piece_kits.py`. Check: `tools/f1b_check.py` -> this folder (42 min on the free runner).

## Numbers (people with feet on the pitch, per frame; test frames not used for fitting)

| | team A | team B | neither | piece vs match: same team / swapped |
|---|---|---|---|---|
| AIK, 3 frames per piece (60) — match model | 4.67 | 2.80 | 4.30 | |
| AIK — piece models | 5.07 | 3.63 | 3.07 | 307 / 78 (20% swapped) |
| AIK, F1's clip-window frames (20) — match / piece | 4.35 / 4.35 | 2.05 / 2.05 | 2.75 / 2.75 | pieces 8+9 fell back to the match model |
| AIK clip window — clip's own model | 4.35 | 3.40 | 1.40 | clip vs match: 65 same / 38 swapped |
| SFK-BP, 3 frames per piece (46) — match / piece | 5.85 / 5.96 | 6.17 / 5.96 | 2.39 / 2.50 | 532 / 2 |

AIK pieces: 13 named, 4 flipped (1, 2, 3, 10), 7 fell back (6 "no clear A/B naming", 1 too few people). SFK-BP: all 19 named, none flipped.

## By eye (`*_strips.jpg`, `*_half_moments.jpg`)

- Second half of AIK (pieces 14-21): each piece model is clean — dark kits on the left, whites on the right, same on every row.
- First half of AIK: messy. The flipped pieces 1-3 have the WHITE team as A (the flip was wrong); piece 4 made the
  yellow-bib substitutes a team; pieces 6-7 mix dark and white in team A. 7 pieces could not be named at all.
- Why: the match model itself mixes the two kits on AIK (its own team A strip holds whites; it disagrees with the clip
  model on 38 of 103 people). Naming the pieces after a model that is wrong in places copies its mistakes. Half the
  pitch is in hard shadow; a white shirt in shadow and a dark shirt in sun end up close in colour.
- Fewer "neither" with piece models (4.3 -> 3.1) but part of that is people moved into the wrong team.

## Verdict

No gain that can be trusted: AIK less "neither" but ~20% team swaps against the match model and wrong flips by eye;
SFK-BP unchanged (harmless). Not wired in; no GPU re-tracking started. Next (F1c): read kit colour relative to the
light the player stands in (shadow vs sun, e.g. lightness relative to the grass right around his feet) and name the teams
by dark vs light directly, not after the match model; grade on the same frames + the strip sheet.
