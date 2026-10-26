# N0-26e (26 Oct): our own shot detector checked on pro data and against Veo

26 Oct nightly card: nothing worse than 25 Oct.

## What the app shows
`stats.metrics.shots` (shot map, shot KPIs, goals) comes from Veo's highlight list when a match has one. Without it
(a new upload with no Veo zip) the app shows **our own detector** (`ipanema/metrics.shots`): the ball suddenly faster
than 16 m/s, within 15 deg of the goal the team on the ball attacks, from within 30 m, and within 3 s the ball crosses
the goal mouth or play stops. On our matches it is only logged against Veo (`metrics.shots_detected`). Never checked on
pro data before.

## Two faults found
1. **'Play stops' could never fire.** Since E4 (29 Sep) the state handed to the detector is possession_simple's, which
   has no 'dead' state. So only a ball whose flat track crosses the goal mouth counted: no saved, blocked or wide shot
   was ever found.
2. **No shot was ever a goal.** A goal needed the next restart to have kind 'unknown' at the centre spot, but the
   restart step calls a centre-spot restart 'stoppage' ('unknown' only when there is no ball spot). The 18 Oct note
   (N0-18b) knew and left it.

## Metrica pro data (`tools/n0_26e_shots.py` -> `metrica_shots.json`)
48 labelled shots in 4 halves (8 goals). Ours on the pipeline's own inputs, perfect ball + our 'streak' / 'flicker'
ball noise (2 seeds each). A true shot matches one of ours of the same team within 2 s.

| rule | ball | ours / truth | found | ours real | goals flagged right (wrong) | outcome right |
|---|---|---|---|---|---|---|
| old | perfect | 14 / 48 | 23% | 79% | 0 of 8 (0) | 3/11 |
| | streak | 18 / 96 | 16% | 83% | 0 of 16 (0) | 3/15 |
| | flicker (like our clips) | 38 / 96 | 34% | 87% | 0 of 16 (0) | 7/33 |
| **new default: goal = first restart within 90 s from the centre spot** | perfect | 14 / 48 | 23% | 79% | **2 of 8 (1)** | 5/11 |
| | streak | 18 / 96 | 16% | 83% | **5 of 16 (1)** | 8/15 |
| | flicker | 38 / 96 | 34% | 87% | **9 of 16 (1)** | 16/33 |
| option IPANEMA_SHOT_END=stops (stoppages end a shot too) | perfect | 21 / 48 | 29% | 67% | 2 of 8 | 5/14 |
| | streak | 40 / 96 | 21% | 50% | 5 of 16 | 9/20 |
| | flicker | 84 / 96 | 52% | 60% | 9 of 16 | 20/50 |

- The goal fix changes nothing but the outcome label and is better on every row -> **on by default**
  (IPANEMA_SHOT_GOAL=unknown = old). On Metrica the kick-off comes 55-76 s after a goal, hence the 90 s window.
- The stoppage end finds more shots but, with noisy balls, many more fakes (real 83 -> 50%, 87 -> 60%) -> **option only**
  (IPANEMA_SHOT_END=stops; run.py now hands it the stoppages so the switch works).
  Also tried in the lab, not kept: aim at the goal mouth instead of 15 deg (perfect 31% / 83%, noisy same as 'stops'),
  team from the last 1.5 s, team = whoever attacks the goal the ball flies at (fakes explode with noise: 1.7-1.9x count, ~35% real).
- Why the rest are missed (perfect ball, 34 misses): angle > 15 deg to the goal centre 11 (shots at a post, headers),
  nobody on the ball in the 0.6 s before 8 (one-touch finishes from crosses), ball under 16 m/s 7 (headers, placed
  shots), wrong team in the state 4, play goes on after a save 4.
- 'On target' only means the flat ball track crossed the goal mouth: shots over the bar read 'on target' (5 of 11
  matched old-rule shots with a perfect ball were really off target). Not changed.

## Our footage
- **App clips** (`tools/n0_26e_clips.py` -> `clips.json`, exact inputs): the default changes nothing (SFK-BP 1 shot,
  AIK 0; no centre restart after either). The stops option would add 1 on SFK-BP (231 s, 'saved or blocked'); Veo has
  no shot in that clip. AIK clip: Veo has 3 shots + AIK's goal, our detector finds none with any rule.
- **Full matches** (`tools/n0_26e_full.py` -> `full_matches.json`, the runs' own `shots_detected` = old rule) vs Veo's
  shots in the same files: **8 of 39 Veo shots found, 10 of our 28 near a Veo shot** (SFK-BP 1/7 and 2/8, AIK 2/16 and
  2/4, Vallentuna 5/16 and 6/16). Veo lists shots on goal only, so 'real' is a lower bound; still, on our footage our
  detector misses ~4 in 5 shots and most of what it shows can't be matched -> question under Waiting for Daniel
  (show nothing instead of our own shots when a match has no Veo list?).
- Also fixed (log line only, no app number): `veo.score_detector` matched within +-10 s of Veo's time, but that time is
  the START of the highlight clip and the shot comes 5-25 s later, so it said 0 found / 0 real on all 3 matches; now
  2 s before to 30 s after.

## Tests
`tests/test_n0_26e_shots.py` (7): goal from a centre restart (and the old rule), no goal when the next restart is
elsewhere or too late, defaults, saved shot needs the stops option (and a stoppage already running doesn't count),
compute hands the stops through, score window. Updated: `test_metrics_shots_runs.py` (pinned 'never a goal' -> goal),
`test_n0_26a_pressmap.py` (run.py call text). Suite 676 pass; 3 fail on this box with or without the change (torch /
ultralytics / supervision not installed).

Reaches the app at the next run (only for matches without a Veo list). $0, nothing on Modal.
