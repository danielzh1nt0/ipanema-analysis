# N0-26b (26 Oct, worker, local + free runner, $0, nothing on Modal): which way does each team attack? Kick-off line-ups

**Card:** 26 Oct nightly says nothing worse than 25 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## Why
The full AIK match processed on the free pipeline (25 Oct, `results/kaggle/full_match_52d795-join`) flags its goals:
after 5 of 7 goals "the scoring team kicks off". The accuracy status of 25 Oct read that as "ownership is wrong around
restarts". Looking closer, the direction itself is in doubt: the run's own direction guess is a coin flip
(confidence 0.01), so it uses `periods/p15u-vs-aik-2026-09-21-bd09.json` (`attack_right_A: true`, by eye 4 Oct:
dark SFK attacks right in the first half), but the keepers in the same run say the opposite.

## A rule that does not guess: kick-off line-ups
At a kick-off every player must stand in his own half, so the half a team lines up in is the half it **defends**.
New `possession.kickoff_spells` / `direction_from_kickoffs`: frames where the ball is unseen or within 15 m of the
centre spot, each team has >= 4 outfield players in the picture more than 1.5 m from halfway, >= 85% of each team on
opposite sides, held >= 1 s.

**Metrica pro data** (`tools/n0_26b_metrica.py`, `metrica.json`; 4 halves, perfect positions + our follow-cam view /
missing players / jitter / ball noise, 3 seeds): **16 of 16 halves right, 51 of 51 spells vote right, 0 wrong**;
26 of 28 kick-offs found under our noise. Settings swept (ball radius 9/15/30/none, split 0.8-0.9, 3-5 players,
hold 1-2 s): not one wrong spell in any setting.

**Our exports** (`tools/n0_26b_kickoffs.py`, `exports.json`, and the new upload check on the app copies):

| match | by-eye file says A attacks | kick-off line-ups say |
|---|---|---|
| Vallentuna (Kaggle join 7) | right | right, 3 of 3 |
| SFK-BP first half (app copy) | right | right, 2 of 2 |
| **AIK full match (free pipeline)** | **right** | **left, 3 of 3** |
| 5-min clips | - | no kick-off in them |

## By eye (free runner, blind sheets of the full video, `results/free/n0-26b/`)
- First-half kick-off (428-446 s, `ko1_02.jpg`): **white AIK lined up in the LEFT half, dark SFK in the right half**.
- Second-half kick-off (3984-4005 s, `ko2_01.jpg`): ends swapped as they should - dark left, white right.
- Kick-off after the 3128 s goal (`full/g3128_t3162..3168.jpg`): again white left, dark right, a dark player at the ball.

So in the first half dark SFK attacks **left**; the file's `true` is the wrong way round for this match. All three
line-ups agree with the pipeline's frames, so the frames' coordinates are right.

What I could NOT settle: where and by whom each goal was scored. Play before the 3128 s goal is at the left end and
before the 6268 s (84', AIK 1-6) goal at the right end, which fits the file, not the line-ups - but the 640 px sheets do
not show the ball going in or who celebrates. With the line-ups fixed by the laws, either those goal moments are not
where the clip times point, or the by-eye goal teams were judged from the wrong end. The goal teams and the 4 Oct
reasoning need a look at the Veo clips themselves -> Waiting for Daniel.

Also: the goals check's "who kicks off" picked the wrong moment (e.g. 2800 s: the ball rolling across the centre circle
before the real kick-off at ~2823 s), so its 5 flags say little; left as is.

## Changes (on by default; `IPANEMA_KO_DIR=0` = old)
- `run.py`: when there is no by-eye direction, kick-off line-ups now come before the ball-drift / keepers guesses;
  when there is one and the line-ups disagree, the log says WARNING.
- `uploadcheck.check_direction` (new row, part `direction`): FLAG when the export's direction disagrees with the
  line-ups in its frames. On the AIK free-pipeline export it flags (3 of 3), the others pass or can't tell.
- Nothing in `periods/` changed: flipping AIK's direction mirrors field tilt, thirds, forward passes, high turnovers
  and the goal ends of that match - Daniel's call (Waiting for Daniel).

## Tests
`tests/test_n0_26b_kickoff_direction.py` (8). Full local suite: 659 pass, 3 fail only for modules missing here
(supervision/torch, as before).
