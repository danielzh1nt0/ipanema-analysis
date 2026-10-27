# N0-27b — pass quality / open and blocked lanes (27 Oct, worker, local + free runner $0)

Nightly card: no 27 Oct card by 04:20 UTC -> backup nightly pushed (free runner); see BACKLOG line for its result.

**What it is:** `passes[].lane_open / quality`, `players[].pass_quality {good, risky_completed, bad_lost, execution_error}`
(the app's "risky passes" = `risky_completed`), events `pass_risky` / `pass_bad` (hidden), the open/blocked lanes in each frame.
Rule (unchanged): at the passer's last frame on the ball, the lane to the receiver is blocked if an opponent stands within
1.5 m of the straight line. Never checked before.

## 1. The rule on perfect inputs (Metrica, all 22 players) - `tools/n0_27b_lanes.py --part 1` -> `metrica_lanes_part1*.json`

1,726 completed passes, 256 cut out (intended receiver = team-mate nearest the cut-out spot; `_line` = the mate in the
direction of the cut-out spot, checks the bias of picking a mate next to the interceptor).

| lane rule | completed called blocked | cut-out called blocked | cut out when blocked | cut out when open | lift |
|---|---|---|---|---|---|
| **app (1.5 m)** | 13.9% | 49.2% (line 54.6%) | 34% | 8.0% | **4.3** (line 5.2) |
| 1.0 m | 8.6% | 36% | 38% | 9.4% | 4.1 |
| 2.5 m | 25% | 67% | 28% | 6.2% | 4.6 |
| widen 0.1 m per m | 19.8% | 65% (line 75%) | 33% | 6.1% | 5.4 (line 8.3) |
| widen 0.2 m per m | 27.8% | 76% | 29% | 4.7% | 6.1 |

The rule means something on real football (unlike 'a better pass was open', N0-27a): a blocked lane is cut out ~4x as often.
Same on all 4 halves (lift 3.7-5.2). A lane that widens with the distance is better still - not tried through our inputs.

## 2. Our inputs vs the rule on perfect inputs - `--part 2` -> `metrica_lanes_part2.json` (28 runs)

| our inputs | lane agrees per matched pass | risky share of completed (truth 12-16%) | blocked -> lost lift (ours) | lost-pass quality agrees |
|---|---|---|---|---|
| perfect ball, all players | 95-97% | 10-13% | 2.1-2.3 | 64-86% |
| flicker ball (like our clips) | 88-93% | 17-22% | 1.1-1.6 | 50-83% |
| camera views | 88-94% | 7-11% | 1.2-2.5 | 33-83% |
| views + flicker | 84-90% | 14-18% | 1.0-1.4 | 44-71% |

## Fault found and fixed

A lost pass's `to` is the opponent who won it, never in the passer's lanes, so **a lost pass never got a lane or a quality**:
`bad_lost` / `execution_error` were ~always 0 (old partial run: ours 0 / 0 vs truth 33 / 33 per half; app copies 0-3 of
25-77 lost passes). New default (`IPANEMA_LOST_LANE=mate`, `old` = old): intended receiver = the passer's team-mate in reach
nearest where the ball was won (`passes[].intended_to`). Completed passes, risky passes and 'better option' are unchanged
(checked on both app clips: identical; a lost pass gets no 'better option', as before).
App clips (`tools/n0_27b_clips.py` -> `clips.json`): SFK-BP lost passes with a quality 0 -> 13 of 13 (bad_lost 7, execution_error 6),
AIK 2 -> 10 of 10.

## By eye - blind sheets of all 23 lost passes that now get a quality (free runner, `tools/n0_27b_frames.py`, results/free/n0-27b)

Graded before opening the app's label: 8 clear real lost passes, 5 not a pass (ball stays with the team, a duel, a camera
jump to a spectator), 10 can't tell. Of the 8 real ones the green 'intended mate' is plausible 7 times. The app's lane
(blocked / open) agrees with my eye 4 of 7 (3 times it said open where an opponent stood in the lane or on the receiver).

## Result

Pass quality holds on pro data (the rule works); through our ball flicker it weakens a lot (lift 1.1-1.6), so it stays
as is: `risky` unchanged, `pass_bad` events stay hidden. The lost-pass labels now exist instead of always 0, but on our
footage only ~1 in 3 of them is clearly a real lost pass -> keep `bad_lost` / `execution_error` out of the app until
M2. tests/test_n0_27b_lanes.py (6).
