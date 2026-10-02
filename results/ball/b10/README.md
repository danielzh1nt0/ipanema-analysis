# B10 - still-ball handling before the picker (2 Oct, free, local)

**Result: no gain. Not switched on.** Picker unchanged (AIK 30/39, SFK-BP 29/34, B4 284/285).

## What was built
- `ipanema/ball.py`: `still_spells` (stretches where the picked ball stays within N px, camera pan removed, for > 1 s),
  `still_prior` (during a spell, and onwards while weak guesses keep showing at the spot, the still spot gets a strong
  guess and other guesses within 2 m of a player are discounted), `pick_still` (pick, add the prior, pick again).
- `tools/b10lab.py` (shared loader `tools/b9lab_lib.py`), test `tests/test_b10_still_prior.py`.
- Graded like B9 on the exact app inputs of both clips: 48 settings (still 10/20 px, 1/1.5 s, boost 0.5/0.8,
  discount 1/0.5/0.25, extend on/off). All numbers in `results/ball/b10lab.json`.

## Numbers
| | AIK /39 | SFK-BP /34 | B4 /285 | real passes kept /20 | fake passes kept /8 | state flips (5 min) |
|---|---|---|---|---|---|---|
| picker today | 30 | 29 | 284 | 20 | 8 | 136 |
| still spells only (no extend) - every setting | 30 | 29 | 284 | 20 | 8 | 136 |
| extend, discount 0.5 (best) | 28-30 | 29 | 283-284 | 20 | 7 | 134-138 |
| extend, discount 0.25 | 27-29 | 29 | 280-284 | 19-20 | 7 | 132-138 |

## Why it does not help
- The fake passes sit where the picker never stays still: in the dead-ball stretch (10.3-13 s, 4 of the 8 fakes) the pick
  jumps between 4-6 spots that all have strong, steady guesses (0.5-0.99). No 1-s still spell is found there at all, so
  there is nothing to lock. Where spells ARE found the picker already stays put, so locking changes nothing.
- 3 of the 18 SFK-BP spells are the spare ball at the goal post (145, 175, 250 s): extending a still spell holds on to it.
- Extending costs AIK 1-3 ball moments and removes at most 1 fake pass.
- Same conclusion as S8: the finder cannot tell the resting ball from marks and shoes nearby. The fix has to come from
  the finder seeing balls at feet better (S8 (a), Kaggle run with extra at-feet crops), not from picker rules.

No pictures made: the clip video is not on this machine; the grading uses the existing by-eye keys.
