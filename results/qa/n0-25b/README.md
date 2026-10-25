# N0-25b (25 Oct, worker, local + free runner $0, nothing on Modal): the team heat map

Card: no 25 Oct card at 04:20 UTC -> backup nightly pushed (triggers/nightly.txt); the 25 Oct card then came in:
nothing worse than 24 Oct (tests 167/167). M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

The app's team heat map (UI contract: SHOW, Beta) had never been checked. The export has no team map; the contract
tells the app to build one from the frames' players or by summing `stats.heatmaps` (one grid per track, each
normalised to 1). Our camera shows ~30% of the pitch around the ball, so both lean to where the ball is.

## Metrica pro data (truth = all 10 outfield players, every team attacking right)

`tools/n0_25b_heat.py`: each Metrica moment seen through a real camera view of our app clips (slid onto Metrica's
ball) with our track breaks and jitter (the N0-22e machinery). 2 games x 2 clips x 2 seeds x 2 teams = 16 maps.
Score: TV = how much of the map is in the wrong place (0 = same, 100 = no overlap); thirds = average error of the
own / middle / attacking third shares.

| rule | TV mean (range) | thirds off (worst) | mean x off |
|---|---|---|---|
| seen players in the frames (what the app can build now) | 12.5% (9-16) | 4.4 pts (5.9) | -0.6 m |
| sum of per-track grids | 12.2% (9-15) | 4.5 pts (6.4) | -0.9 m |
| exposure-corrected (per cell: seen / time in view) | 22.0% | 13.3 pts | -1.8 m (rejected) |
| hold 5 s / 10 s | 11.1% / 11.0% | 3.6 / 3.0 pts | -1.2 / -1.0 m |
| **hold 20 s (new `teams[].heatmap`)** | **11.0% (9-14)** | **2.2 pts (4.5)** | -0.9 m |
| hold 30 / 60 s | 11.2% / 11.7% | 1.8 / 1.9 pts | -0.9 m |

Better than the seen map on all 16 maps (TV and thirds). For scale (`baselines.json`): our breaks + jitter alone with
a whole-pitch camera = 1.7%; the two pro teams' true maps differ by 7-13%. So the map's broad shape (which third,
which side) holds; single-cell detail does not - the camera error is as big as the difference between two teams.
Pictures: `metrica_maps.png`.

## New export field (on by default; `IPANEMA_TEAM_HEAT_HOLD_S=0` = seen players only)

`stats.teams[].heatmap = {grid (12 x 8, same layout as stats.heatmaps, sums to 1), hold_s, frames, thirds_pct
[own, middle, attacking]}` from `analytics.team_heatmaps` (keepers out, edge rule of N0-22e: no held spot within
1.5 m of a line). Nothing else changes. The app shows it only after a Lovable change (Waiting for Daniel).

## Our clips (repo copies of the app's exports, `clips.json`, `clips_maps.png`)

Old vs new map differ by 6-17% (TV); thirds shares move 0-3 pts (SFK-BP A own third 36 -> 33, AIK A 13 -> 16).
By eye the new maps look like the old ones, a little more spread up the pitch.

## Found on the way: people off the pitch counted as players

Our maps put 2-15% of a team's time within 3 m of a touchline (pro players ~1-2%). The 12 people per clip with the
most touchline time were cut as sheets on the free runner (`tools/n0_25b_moments.py`, `tools/n0_25b_frames.py` ->
`results/free/n0-25b/`), mixed with 6 random mid-pitch players each, graded blind (`grades.json`), then the key opened:

| clip | touchline people: mostly not a player | tiles: player / not / can't tell | mid-pitch controls: not a player |
|---|---|---|---|
| SFK-BP | 4 of 12 (bench staff, people seated on the far bench) | 31 / 17 / 0 | 0 of 6 |
| AIK | 9 of 12 (spectators behind a touchline, one bag) | 10 / 29 / 9 | 0 of 6 |

Most project to 0-1.5 m inside the far line (clipped at the line), so the same spots also enter possession, pressure
and the shape numbers. Wide players also live there (SFK-BP 6 of 12 were real wingers / throw-ins), so a plain
position cut would remove real players. -> new queue item TOUCHLINE.

Tests: `tests/test_n0_25b_team_heat.py` (6). Suite 626 pass. Tools: `n0_25b_heat.py`, `n0_25b_baselines.py`,
`n0_25b_plot.py`, `n0_25b_clips.py`, `n0_25b_moments.py`, `n0_25b_frames.py`.
