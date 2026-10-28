# N0-28e (28 Oct, worker, local $0, nothing on Modal): pass network + pass length on pro data

Card: the 28 Oct card's 'worse' rows (SFK-BP dark 8 -> 7, Reymersholm dark 6 -> 5, Djursholm not run) were already
explained and fixed today (N0-28c: the old numbers were frozen 28 Sep copies; N0-28d: the runner's OpenCV pin).
The 29 Oct card is not out yet. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before)
- `stats.metrics.pass_network` - who passes to whom (edges with counts) and where each player sits (node = mean of the
  spots where he played / received passes). In the app's data it is still track pieces, not players (M2), so it is
  hidden there. This check answers: **once M2 gives one id per player, will the network be right?**
- `passes[].length_m` - the '9.0 m, gain 6.1 m' on every pass event in the timeline (gain was checked in N0-23c).

Metrica's 2 pro games, 4 halves; truth = their labelled passes (From / To, ball start / end). Ours = the pipeline's
passes on Metrica's positions with a perfect ball and our two kinds of ball noise ('streak', 'flicker', 2 seeds each)
= 20 runs, 40 team-halves. `tools/n0_28e_passnet.py` -> `metrica_passnet.json`.

## Result: the network holds; lengths hold, long balls read short

| | perfect ball | streak | flicker |
|---|---|---|---|
| network overlap with the truth (1 = identical) | **0.70** (min 0.67) | 0.63 (0.56) | 0.57 (0.50) |
| same, a network that only knows who is busy (no pair info) | 0.43 | 0.42 | 0.42 |
| truth's 10 busiest pairs among our 10 busiest | 7 | 6 | 6.5 |
| rank agreement of pair counts | 0.74 | 0.61 | 0.57 |
| THE busiest pair right | 6/8 | 9/16 | 4/16 |
| node position off (median) | 3.1 m | 3.9 m | 4.3 m |
| passes in the network vs truth | 1.03x | 0.99x | 1.11x |
| pass length off (median, matched passes) | 0.55 m | 1.4 m | 0.67 m |
| pass length within 5 m | 94% | 90% | 88% |
| long balls (>= 30 m) we also call long | 69% | 61% | 45% |
| our long balls that are really long | 100% | 85% | 100% |

For scale: a team's own two halves (truth vs truth) overlap only 0.28-0.40, so our network is much closer to the
truth than a team's network is to itself 45 minutes later.

By eye (`passnet_g2h1_A.png`, game 2 first half, team A): truth, perfect ball and flicker ball show the same picture -
the 4-7-8 triangle, 6-4 as the main channel, 1-5 and 9-8; nodes within a few metres.

So: **once M2 exists the pass network can be shown** (overlap well above 'who is busy', top pairs mostly right,
nodes 3-4 m off). Don't headline 'most common pass': with our kind of ball flicker the single busiest pair is right
only 1 in 4. The pass length in the event subtitles is fine (0.5-1.4 m off).

Long balls: about 1 in 3 is cut short (even with a perfect ball). Looked at all 32 long passes of game 2 first half:
when the ball flies over / past a team-mate in 2D, he becomes a contact and the 45 m ball becomes a 12 m pass to him
(the rule has `mate_sandwich_s` for this, off since S1); 8 long passes are not matched at all. A fix would change the
pass count everywhere - not done here; a 'long balls' stat should not be added until it is.

## No pipeline change
Nothing in the app or the export changed. Nothing new for Daniel to decide (the network is already hidden until M2).

## Files
- `tools/n0_28e_passnet.py` (lab), `tools/n0_28e_plot.py` (picture), `metrica_passnet.json` (summary + 20 runs),
  `passnet_g2h1_A.png`.
- `tests/test_n0_28e_passnet.py` (6): edges count completed passes only, the lab reads exactly what
  `metrics.pass_network` writes, overlap / top-10 / busiest / node scores, the 'who is busy' baseline, length matching,
  Metrica row parsing. Full suite 743 pass (supervision 0.30 installed on this box; without it `test_tiles_detect` fails
  at import, as before).
