# TOUCHLINE (25 Oct, worker, local + free runner $0, nothing on Modal): people standing by a touchline

Card: 25 Oct card nothing worse than 24 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## The problem (from N0-25b)

People just behind the far touchline (spectators, bench staff, seated subs, bags) project 0-1.5 m inside the line and
are tracked as players (AIK clip 9 of the 12 people with most touchline time, SFK-BP 4 of 12). They feed possession,
pressure, shapes and heat maps.

## First finding: half of it is already fixed in the code

The app clips were exported on 4 Oct. `tracking.clean()` got a track rule on 9 Oct (a track that is >= 70% of its rows
within 1 m of a line is dropped). Applied to the clips' saved inputs it removes 17 of the 46 'not a player' tiles graded
in N0-25b (and 2 of 85 player tiles) - AIK 2.5% of all player rows, SFK-BP 0.5%. It reaches the app at the next run.

What it misses: track ids slide between a spectator and a player, so a track is only partly by the line, and many
bystanders stand 1-3 m 'inside' it.

## New rule (on by default, `IPANEMA_TOUCHLINE_S=0` = old)

`tracking.touchline_rows`, step 1c of `clean()`: cut each track into stretches (break at a gap > 0.5 s or a jump > 2 m in
0.1 s = another person under the same id); a stretch that lasts >= 6 s, is never deeper than 3 m from a side touchline
and stands (median speed of the positions smoothed 0.8 s <= 1 m/s) is dropped. Per stretch, not per track.

## Grading

Lab (`tools/touchline_lab.py`, `lab.json`, 80 settings): the 36 people graded blind in N0-25b (tiles: 46 not a player,
85 player), Metrica pro tracking cut like our tracks (every row a real player), rows dropped on the clips.

| rule (after the 9 Oct rule) | not-a-player tiles removed | player tiles lost | Metrica rows lost (worst half) |
|---|---|---|---|
| 9 Oct rule alone | 17 / 46 | 2 / 85 | - |
| + 4 s / 3.5 m / 1 m/s | 36 / 46 | 6 / 85 | 0.51% |
| + 6 s / 3.5 m / 1 m/s | 32 / 46 | 3 / 85 | 0.21% |
| **+ 6 s / 3 m / 1 m/s (new default)** | **29 / 46** | **3 / 85** | **0.20%** |
| + 8 s / 3 m / 1 m/s | 22 / 46 | 2 / 85 | 0.14% |

On Metrica about half of the few rows lost are while the ball is out of play; ~1% with the ball within 5 m.

Fresh blind sample (`tools/touchline_moments.py`, sheets on the free runner `tools/touchline_frames.py` ->
`results/free/touchline/`): the app clips have only 9 stretches the rule fires on, all already graded, so the sample came
from the repo's full-match exports - Vallentuna (a third ground), SFK-BP full (minus the clip window) and AIK full. Drawn
at 3.5 m: 17 stretches the rule drops + 8 it keeps near the line, graded blind (`grades.json`) before opening the key:

| | not a player | player |
|---|---|---|
| dropped at 3.5 m | 15 (kids with a ball behind the line, bench people, coaches, a sub waiting, bags, spectators by the spare goal) | 2 (a player at a stoppage, a wide player standing; both 3.4 m deep) |
| dropped at 3 m (new default) | 11 | 0 |
| kept by the line (moving > 1 m/s) | 4 | 4 |

So at 3 m: nothing real lost on the fresh sample, 11 bystanders gone. Missed: bystanders that 'move' faster than 1 m/s
(the camera pan makes still people drift a little in metres) - a speed limit high enough for them also takes wingers.

## Effect on the app clips (`clips_effect.json`)

Share of player rows within 3 m of a touchline: AIK 8.2% as exported -> 5.9% with the 9 Oct rule -> 5.0% with the new
rule; SFK-BP 4.3 -> 3.8 -> 3.5% (pros ~1-2%). New rule drops 0.97% / 0.37% of rows. Reaches the app at the next run.

Tests: `tests/test_touchline.py` (6). Tools: `touchline_lab.py`, `touchline_moments.py`, `touchline_frames.py`.
