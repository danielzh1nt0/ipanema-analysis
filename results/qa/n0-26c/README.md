# N0-26c (26 Oct): per-player touches checked on pro data and by eye

26 Oct nightly card: nothing worse than 25 Oct.

## What the app shows
`stats.json players[].touches` (the app's player table reads it). It was counted from every run of frames where the
player is the "carrier" (nearest to the ball, within 2.5 m), a new touch after a gap > 0.4 s. A ball flying past, a
1-frame ball pick next to someone, a duel - each made a touch. Never checked before.

## Metrica pro data (`tools/n0_26c_touches.py` -> `metrica_touches.json`)
Truth: every spell on the ball ends with one PASS, SHOT or BALL LOST by that player, so real touches per player =
those events. Ours = the pipeline on Metrica's positions, perfect ball + our 'streak' / 'flicker' ball noise (2 seeds),
4 halves = 20 runs, 40 team-halves. Players with >= 5 real touches.

| rule | count vs truth | player off (median) | rank order (Spearman, median) | top player right |
|---|---|---|---|---|
| old (carrier runs) | 1.65-4.00x | 172% (59-290) | 0.60 | 14/40 |
| **new: cleaned contacts (as passes use them, spell state)** | 1.18-1.66x | 29% (18-62) | 0.88 | 28/40 |
| contacts, no state filter | 1.22-2.14x | 44% | 0.82 | 23/40 |
| contacts >= 0.2 s | 1.00-1.48x | 17% | 0.91 | 29/40 |
| contacts >= 0.4 s | 0.79-1.17x | 14% | 0.92 | 23/40 |

New is better on every run. A minimum contact length is better still on Metrica, but NOT used: on our clips most
contacts are short (median 0.25 s), 0.4 s would keep 42 of 108 on the SFK-BP clip.

## Our app clips (`tools/n0_26c_clips.py` -> `clips.json`)
SFK-BP clip (5 min): touches 342 -> 108 (A 184 -> 61, B 158 -> 47); AIK clip: 412 -> 127. For comparison, that clip's
passes are 59 (about 62 real by the by-eye key), so 108 is plausible and 342 is not. 'Players' are still track pieces
(83-91 with a touch per clip), so the per-player figure only means something after M2.

## By eye, blind (`tools/n0_26c_frames.py` on the free runner -> results/free/n0-26c, grades in `grades.json`)
36 sheets (18 per clip, seeded): 24 old touches the new rule drops, 12 it keeps. Graded without knowing which:
does the marked player play / control the ball around the 3rd tile?

| | real | not | can't tell |
|---|---|---|---|
| dropped (24) | 6 | 13 | 5 |
| kept (12) | 4 | 4 | 4 |

- What it drops is mostly not a touch (13 of 19 clear): the ball nowhere near, another player has it, the referee
  (#34), a spare ball by the fence (#26). The 6 real ones are mostly 1-frame flashes of a player who has the ball just
  before / after (often already counted in his neighbouring contact).
- What it keeps is real only about half the time (4 of 8 clear): the same ball and identity problems that limit passes.
- Rough share of counted touches that are real: old ~4 in 10, new ~5 in 10.

## Decision
New is the default (`IPANEMA_TOUCHES=old` = old count; `IPANEMA_TOUCH_MIN_S` = minimum contact, default 0). Reaches the
app at the next run. Per-player touches stay weak on our footage until M2 (one track per player) and a steadier ball:
whether the app should keep showing them is a question for Daniel (BACKLOG, Waiting for Daniel).

Code: `analytics.touch_episodes` (split out of `passes()` unchanged), `analytics.touch_counts`, `analytics.stats(...,
episodes_=)`, `run.py` passes the same contacts the passes use. Tests: `tests/test_n0_26c_touches.py` (6).
