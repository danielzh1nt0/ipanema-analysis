# N0-29f (29 Oct, worker, local $0, nothing on Modal): the app's possession % for a whole half

Card: 29 Oct card says nothing worse than 28 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked
`stats.json teams[].possession_pct` - the KPI 'Possession' and the summary line, the app's headline number. It is
A's share of the frames where the raw possession state says A or B (loose frames left out). The nightly only says
"worst half 8 pts off" on pro data; nobody had looked at why, or at other ways to count it.

## How (no clicking)
Metrica's 2 pro games, 4 halves. Truth per frame = team of the last on-ball event, ball out / a foul until the restart =
nobody's (N0-25d's truth). Ours = the pipeline's steps on Metrica's positions, perfect ball, our clips' 'flicker' ball,
'streak' ball and 'veo' (65% of players, balls missing / wrong), 2 seeds. 9 ways of counting from the same captured
state. `tools/n0_29f_possession.py` -> `metrica_possession.json`, picture `possession_error.png`.

## Result: the app's rule is fine; the nightly's 8 pts is mostly Metrica's missing ball

| ball | whole half, mean / worst off (pts) | where Metrica has a ball, mean / worst off |
|---|---|---|
| perfect | 2.2 / 4.6 | 2.5 / 4.1 |
| flicker (like our clips) | 5.7 / 17.1 | **1.2 / 1.8** |
| streak | 5.1 / 10.6 | 3.1 / 7.4 |
| veo | 10.3 / 23.1 | **1.9 / 2.9** |

- Metrica blanks its ball for 28-55% of every half, mostly in long in-play stretches (up to 130 s). With no ball the
  state keeps the last team, and with a noisy ball that last frame is a coin toss - so the same half reads 39% or 59%
  depending only on the noise seed (Game 2, half 2, flicker). That is what the whole-half error measures.
- Our footage never has this: the picker always leaves a ball. In the in-period frames of the repo exports there are
  0 ball-less runs of 2 s+ on SFK-BP (full half and clip) and AIK, and 1 of 2.6 s on Vallentuna.
- On the frames where Metrica does have a ball - like ours - the app's count is 1-3 pts off on average with any ball
  model (worst 7 with streak noise), and the right size of error for Vallentuna's blind by-eye check (app 43%, by eye
  42-44%).

## Tried, not switched on
Counting from the spell state, crediting loose frames to the last team, leaving stoppages out, leaving ball gaps of
1/2/5 s out, filling gaps from the spell state. Ball-gap rules make the whole-half Metrica score look much better with
noise (veo 10.3 -> 3.9) but worse with a perfect ball (2.2 -> 3.0) and change nothing on our footage (no gaps), so the
app's rule stays. The spell state is not better where the ball is seen (flicker 1.2 -> 2.0).

## Changed: the nightly card says it too
`tools/nightly_quick.py` also scores the possession % on the frames outside Metrica's ball gaps of 2 s+, and the card row
now reads e.g. `possession 8 pts off (4.1 where Metrica has a ball, N0-29f)`. The 'worse than' comparison still reads
the first number (unchanged). Dry run of the whole nightly script locally: same tests / ball / passes / possession
numbers as the 29 Oct card (15%, 8 pts, 22/34), new number 4.1 (`nightly_quick_dryrun.json`).

Tests: `tests/test_n0_29f_possession.py` (7): share leaves loose frames out, fill, gap mask, the variants on a stand-in
state, the truth on the same frames, the app's rule = 'raw', and the card row still parses with the new note (and
without it on an old quick.json). Full suite 776 pass (3 files need `supervision`, not on this machine). $0.
