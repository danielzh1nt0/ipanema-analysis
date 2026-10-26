# N0-26d (26 Oct): per-player time on the ball and passes received checked on pro data

26 Oct nightly card: nothing worse than 25 Oct.

## What the app shows
`stats.json players[].time_on_ball_s` (and `time_under_pressure_s`, the part of it with an opponent within 2 m) counted
every frame the player was the "carrier" (nearest to the ball, within 2.5 m). `players[].passes_received` = completed
passes whose receiver is the player. Neither checked on pro data before.

## Metrica pro data (`tools/n0_26d_onball.py` -> `metrica_onball.json`)
Truth from Metrica's labelled events: time on the ball = from the ball reaching the player (end of a pass to him, or his
recovery / set piece / carry) to him playing it (his pass / shot / ball lost), one-touch = 0 s, capped at 20 s. Passes
received = passes to him. 4 halves x (perfect ball + streak and flicker noise, 2 seeds) = 20 runs, 40 team-halves,
players with >= 5 real touches.

**Passes received - holds.** Team total 0.87-1.30x the truth (median 0.99-1.11), a player median 12% off (perfect),
16% (streak), 21% (flicker); rank order 0.81-0.92. No change.

**Time on the ball** (medians over the team-halves; ratio = ours / truth total)

| rule | ball | ratio | player off | rank order | share of team off |
|---|---|---|---|---|---|
| old (every carrier frame) | perfect | 1.59x | 63% | 0.96 | 0.9 pts |
| | streak | 0.85x | 23% | 0.92 | 1.4 |
| | flicker (like our clips) | 1.66x | 71% | 0.85 | 1.4 |
| state (only while the spell state gives his team the ball) | perfect | 1.36x | 42% | 0.94 | 0.8 |
| | streak | 0.39x | 64% | 0.85 | 1.2 |
| | flicker | 1.23x | 26% | 0.92 | 1.1 |
| **new: carrier frames inside his cleaned contacts** (the ones touches and passes use) | perfect | 1.50x | 54% | 0.96 | 0.7 |
| | streak | 0.69x | 28% | 0.96 | 1.1 |
| | flicker | 1.27x | 30% | 0.93 | 1.0 |

- The seconds read too high with a clean ball (the 2.5 m 'carrier' circle counts the ball arriving and leaving), so the
  absolute seconds are rough on any rule; who had the ball most (rank, share of the team) is right on every rule.
- New vs old: player error same or better on all 8 perfect-ball and all 16 flicker team-halves (flicker 71 -> 30%),
  worse on 13 of 16 streak ones (23 -> 28%: there the old rule was lucky - its 'too many' frames offset the missing ball); share of team better
  on 37 of 40, rank order same or better on 34 of 40.
- 'state' alone breaks badly with streak noise (0.39x) - not used.

## Our app clips (`tools/n0_26d_clips.py` -> `clips.json`)
| clip (5 min) | team seconds on the ball old -> new | player pieces with time | ... of them with 0 touches |
|---|---|---|---|
| SFK-BP | A 25.0 -> 12.7, B 39.4 -> 27.6 | 155 -> 83 | 72 -> 0 |
| AIK | A 40.7 -> 25.9, B 25.4 -> 15.5 | 159 -> 91 | 68 -> 0 |

The top players barely move (e.g. 3.8 -> 3.6 s, 5.6 -> 5.5 s); what goes is scattered single frames of ~70 track pieces
per clip that, since N0-26c, the app shows with seconds on the ball but 0 touches. The dropped frames are the same
'carrier runs outside a contact' that N0-26c graded blind by eye: 13 of 19 clear ones were not a touch. No new picture
sheets needed. Seen: on the AIK clip team A shows more on-ball seconds than B although B has much more possession
(92 vs 163 s) - a ball / identity problem (M2), not this rule.

## Decision
New is the default (`IPANEMA_ONBALL=old` = old), better on the noise our clips have and on perfect data, and the player
table now agrees with itself (seconds on the ball only for players with a touch). Time under pressure uses the same
frames. Reaches the app at the next run. Per-player figures still mean little until M2 (one track per player).

Code: `analytics.touch_frames`, `analytics.touch_time`, `analytics.stats` (episodes_ given -> on-ball frames = contact
frames). Tests: `tests/test_n0_26d_onball.py` (4). $0, local only, nothing on Modal.
