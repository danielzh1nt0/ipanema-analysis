# N0-22e (22 Oct, worker, local only, $0, nothing on Modal)

**Card 22 Oct:** worse than 21 Oct = nothing (tests 152/152, ball 22/34, pro-data passes 15% / possession 8 pts off).
M2 / PC2 / BALL-EDGE skipped (main session / waits for M2 / needs a Kaggle GPU; this worker runs only local + the free runner).

**What was looked at:** the team shape numbers in `stats.json` (`teams[].block_length_median_m`, `block_width_median_m`,
`def_line_height_median_m`). The accuracy status of 17 Oct lists "block, line height" as never checked by anyone.
They are labelled Beta ("visible players") in the UI contract.

## Finding
Our camera shows only part of the pitch. So the length and width come only from the players in the picture, and they
read short.

How it was measured, with no answer key for our players:
- Truth = Metrica's free pro tracking (all 20 outfield players).
- Each Metrica moment gets a real camera view from our two app clips (the pitch area our camera showed, from the clip's
  calibration), slid so its centre sits on Metrica's ball.
- Our track breaks and jitter are added (new id at every re-detection, like our tracks).
- Tools: `tools/n0_22e_shape_view.py`, `tools/n0_22e_pipeline_check.py`.

Results (2 games x 2 clips x 2 seeds, teams' medians as % of the truth; `metrica_pipeline.json`):

| rule | block length | block width | line height (m off) |
|---|---|---|---|
| old: players in the picture | **68-76%** | **75-81%** | -2.9 .. +1.4 |
| hold 3 s | 85-89% | 86-92% | same |
| **hold 5 s (new default)** | **92-97%** | **90-96%** | same |
| hold 8 s | 100-106% | 94-101% | same |

- The line height (deepest outfield player) was already about right from the players in the picture. Holding makes it
  worse (`metrica_views.json`: about -1.5 m), so it keeps the old rule.
- Rules that only count frames where the camera shows the whole team (`cover`, `seen8`) are worse. They pick compact
  moments, so the length drops to 12-21 m.

## Fix (on by default; `IPANEMA_SHAPE_HOLD_S=0` = old numbers)
`analytics.held_points`, used by `analytics.stats`: when a player leaves the picture, his last seen spot keeps counting
for 5 s. This is the usual "last known position".

Two rules:
- A frame needs at least 3 seen outfield players of that team.
- A spot within 1.5 m of the pitch's edge is not held (`IPANEMA_SHAPE_EDGE_M`). On our copies 16-18% of the held spots
  sat on a touchline or goal line, while only 3-8% of the seen spots did, and 0.7% of Metrica's pro players. They are
  people at the line, or feet cut off by the edge of the picture. With the rule, 0% of held spots are on the edge, and
  the Metrica numbers are unchanged.

Each team row gets `block_hold_s`. The per-frame hull (`frames[].shape`, drawn on the video) and `metrics.shape_timeline`
are unchanged. They show what the camera sees.

## Repo copies of the app's exports (`tools/n0_22e_app_copies.py`, `app_copies.json`, 10 frames/s)
| match | length A old -> new | B | width A | B | line height |
|---|---|---|---|---|---|
| SFK-BP full 1st half | 26.6 -> 30.4 m | 25.3 -> 28.1 | 32.8 -> 36.1 | 32.1 -> 35.1 | unchanged |
| Vallentuna full | 28.1 -> 34.5 | 25.6 -> 29.1 | 37.6 -> 41.4 | 36.8 -> 39.1 | unchanged |
| AIK (2 of 12 frame files) | 29.6 -> 39.3 | 28.7 -> 38.0 | 38.8 -> 47.9 | 45.6 -> 49.7 | unchanged |

The change is +10-35% on length and +6-23% on width. The Metrica check predicts +30% and +18%.

## By eye (`held_*.png`)
Pitch drawings of 6 random frames per match: filled = in the picture, hollow = held. The bars on top compare the new
block length (solid) with the old one (dotted).
- SFK-BP and Vallentuna: the held spots sit where those team-mates plausibly are, inside the team's area.
- AIK: some held spots lag behind after a fast camera swing (frame 1029: a group of A spots 5 s old on the other side).
  In two frames, people on the near touchline are counted while seen. That is not new, and it is a tracking issue.
- The first version of this drawing showed held clusters on the touchlines. That led to the edge rule.

## Limits
- The real players aren't known. The 5 s choice rests on Metrica seen through our views.
- A team still shows up to ~14 spots in a frame (p90), because broken tracks leave duplicates. Duplicates sit next to
  the real player, so they hardly move the length or width. The Metrica check includes them.
- One track per player (M2) would make this cleaner.
- The app's label "visible players" could become "estimated, whole team" (optional Lovable text change).
- Reaches the app at the next run or join (Modal, Daniel).

## Tests
`tests/test_shape_hold.py` (6):
- held for 5 s, then dropped
- keeper left out
- edge spot not held
- an empty frame clears the held spots
- a frame needs 3 seen players
- stats: block length 35 -> 55 m when the deepest two are out of the picture 2 s in 3, line height unchanged
- the old switch gives exactly the old numbers
- dry run of the Metrica tool

Full suite: 563 pass, 4 skipped (no torch here).
