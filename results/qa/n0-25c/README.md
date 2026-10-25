# N0-25c (25 Oct, worker, local only, $0, nothing on Modal)

**Card 25 Oct:** worse than 24 Oct = nothing.
M2 / PC2 skipped (main session / waits for M2).

**What was looked at:** the app's "shape over time" (`stats.metrics.shape_timeline`). It gives one value per second per
team: block length, width and line height. The app's Phases ribbon and shape-over-time chart read it. Nobody had
checked it against a truth before.

## Finding
On 22 Oct (N0-22e) the Shape tab's team **medians** were fixed: players who just left the picture keep their last spot
for 5 s. The **timeline** was left on the old rule: only the players in the picture, and just 3 of them are enough.
So the chart and the cards disagreed. On the SFK-BP first half the chart sat around 25 m while the card said 30 m.

How it was measured: the same set-up as N0-22e. Metrica's free pro tracking (all outfield players) is the truth. Each
moment is seen through a real camera view from our two app clips, with our track breaks and feet jitter added. Then the
shipped timeline code is run, and every second is compared with the truth at that moment
(`tools/n0_25c_timeline.py`; 2 games x 2 clip views x 2 seeds = 8 runs; `metrica_timeline.json`).

| per second | old (players in the picture, >= 3) | new (held 5 s, >= 5 points) |
|---|---|---|
| block length, median error | **10-12 m** | **2.9-3.7 m** |
| block length vs truth | 0.60-0.67x | 1.00-1.01x |
| block width, median error | 12-14 m | 3.7-4.6 m |
| block width vs truth | 0.67-0.70x | 0.93-0.97x |
| seconds with a value | 87-93% | 73-82% |
| line height | 1.4-2.2 m off (1.05-1.07x) | unchanged |

New is better on all 8 runs. The cost: fewer seconds get a value. A second where the camera shows only 3-4 players of a
team now gets a null instead of a "block" of 3 people. On pro data the new value overshoots the truth by more than
15 m in only 0.2-0.3% of seconds.

Picture `metrica_timeline.png` (5 minutes, game 1): the new line (red) follows the truth (black); the old one (grey)
sits 10-20 m under it and jumps.

## Fix (on by default; `IPANEMA_TIMELINE_HOLD=0` = old)
`metrics.shape_timeline` now gets the pipeline rows. Length and width come from `analytics.held_points`, the same rule as
the team cards. Line height stays as it was (N0-22e: holding makes the line worse). Each second also gets `n_held`.
`IPANEMA_SHAPE_HOLD_S=0` switches both the cards and the timeline back.

## Repo copies of the app's exports (`tools/n0_25c_app_copies.py`, `app_copies.json`)
| match | length A old -> new (timeline median) | card median | seconds with a value |
|---|---|---|---|
| SFK-BP full 1st half | 25.4 -> 30.4 m | 30.4 | 82% -> 79% |
| Vallentuna full | 26.8 -> 34.6 m | 34.5 | 81% -> 78% |
| AIK (2 of 12 frame files) | 27.9 -> 39.2 m | 39.3 | 99% -> 95% |

The chart now agrees with the card. Pictures: `timeline_*.png` (old grey, new red, card dashed).

## By eye: the long seconds (`spikes_*.png`, `spikes.json`)
On our copies, more seconds now read a block longer than 60 m:

| match | old | new |
|---|---|---|
| SFK-BP | 0% | 0.3% |
| Vallentuna | 0.1% | 2.3% |
| AIK | 0.6% | 7.0% |

I drew 6 random ones each for AIK and Vallentuna:
- **6 of 12:** people on or by the touchline, seen or held: bench staff, spectators, bags, the same problem as TOUCHLINE.
- **3 of 12:** held spots left behind after a fast camera swing (lag), or I can't tell.
- **3 of 12:** a plausible stretched team.

Applying today's line rules (the 9 Oct and TOUCHLINE rules, not in these 4 Oct copies) only takes AIK from 7.0% to 6.2%.
These spikes come from our tracking, not from the timeline rule. The card medians have the same inputs; a median hides
them, a per-second chart doesn't. One track per player (M2) and a cleaner touchline are the real fix.

## Limits
- The truth is pro players seen through our camera views, not our own players. Same limit as N0-22e.
- Reaches the app at the next run (Modal, Daniel). Nothing else needs a decision.

## Tests
`tests/test_n0_25c_timeline.py` (7):
- held length 35 -> 55 m when the deepest two are out of the picture; line height unchanged
- old switch gives exactly the old timeline
- the timeline median equals the team card
- too few points: null length but the line stays
- `IPANEMA_SHAPE_HOLD_S=0` gives the old timeline
- `compute()` passes the rows
- env switch
- tool dry run

Full suite: 631 pass, 7 skipped. 3 fail: they need supervision, which isn't on this machine (same as before).
