# N0-26a (26 Oct): pressing over time / pressure map checked on pro data and by eye

26 Oct nightly card: nothing worse than 25 Oct (the card was missing at 04:20, so the backup nightly was pushed first).

## What the app shows
`stats.metrics.pressure_points[]` (ipanema/metrics.pressure_points, every run): one point per second per carrier while an
opponent is within 2 m of him. The carrier is the player nearest the ball (<= 2.5 m); no possession state is used. The app
counts the points per 5 minutes per pressing team ("Pressing over time", lovable_fixes_2026-10-04 #10) and draws them as
the pressure map. This had never been checked.

## Metrica pro data (`tools/n0_26a_pressmap.py` -> `metrica_pressmap.json`, `tools/n0_26a_summary.py` -> `summary.json`)
Truth = the same rule with the real carrier (from Metrica's events) and real positions, only while the ball is in play.
Ours = the pipeline on Metrica's positions with a perfect ball and our two ball noise models (2 seeds), 4 halves.

| rule / ball | count vs truth | our points real | real found | 5-min block error | team on top per 5 min | thirds off |
|---|---|---|---|---|---|---|
| app (old), perfect | 1.34-1.56x | 58% | 83% | 48% | 21/33 | 3.3 pts |
| app (old), streak | 1.73-2.42x | 37% | 71% | 86% | 41/66 | 4.3 pts |
| app (old), flicker | 2.09-2.73x | 37% | 86% | 130% | 45/66 | 4.5 pts |
| spell-state option, perfect | 0.72-1.05x | 77% | 71% | 19% | 26/33 | 5.2 pts |
| spell-state option, streak | 0.39-0.80x | 58% | 36% | 38% | 47/66 | 7.5 pts |
| spell-state option, flicker | 0.90-1.30x | 56% | 64% | 25% | 46/66 | 5.8 pts |
| lab: raw state, perfect / streak / flicker | 0.98 / 0.54 / 1.45x | 76 / 59 / 49% | | 16 / 48 / 42% | 25 / 40 / 40 of 33/66/66 | |

- On pro data the count reads too high (every ball that passes or bounces near a player with an opponent close makes a point)
  and the 5-min chart is noisy. **Where** pressing happens holds: thirds within 3-5 pts of the truth.
- 12x8 grid cells are not meaningful at this many points: half of the truth against the other half is already 49-58%
  misplaced, a random 60% of the truth 21-23% (`grid_misplaced_pct` 18-39% everywhere is at that floor).
- Option built: only count frames where the counted spell state gives the carrier's team the ball
  (`IPANEMA_PRESS_POINTS=spell`). On Metrica it is better on every count row (count, 5-min error, team on top per block).

## Our two app clips, by eye (`tools/n0_26a_clips.py` -> `clips.json`, `moments.json`; free-runner sheets `results/free/n0-26a`)
Old rule: SFK-BP 32 / 31 points (dark / light kit pressing), AIK 32 / 30; the option drops 33 and 30 of them
(SFK-BP 17 / 14, AIK 22 / 10). A seeded 36 (12 dropped + 6 kept per clip) were cut as 6-tile full-resolution crops
and graded blind (`grades.json`, graded before opening which ones the option drops):

| | real pressure on a player on the ball | not | can't tell |
|---|---|---|---|
| points the option DROPS (24) | 18 | 4 | 2 |
| points both keep (12) | 7 | 5 | 0 |

Most dropped points are duels: a player on the ball with an opponent on him, while the spell state (it waits 1.5 s
before it changes team) still names the other team. The false points (both rules) are dead balls / players walking with
the ball pick on empty grass, a ball in flight, or the line on the wrong pair. On our footage the old points are real
about 7 in 10 (25 of 34 graded), much better than the noise models suggest. 3 moments were counted for both teams
within 0.1 s (the carrier hops between the two players of a duel).

## Result
Not better on our footage, so **nothing switched**: the app's rule stays, the option is off (`IPANEMA_PRESS_POINTS=spell`
to try). The chart reads high on pro data, but by eye on our clips most points are real pressures; the team split per
5 minutes is the weak part (per-block team on top right ~2 in 3 even on pro data). Pressing over time is fine as Beta;
the pressure map (where) is the more reliable half.

Tests: `tests/test_n0_26a_pressmap.py` (7). $0, nothing on Modal.
