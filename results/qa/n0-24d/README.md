# N0-24d (24 Oct, worker, local $0, nothing on Modal): pressures and PPDA on pro data

Card: 24 Oct card says nothing worse than 23 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

Never checked against a truth before (both are HIDE in the app):

- `teams[].pressures_applied` - per opponent possession sequence, each of our players who comes within 2 m of their
  carrier counts once (S7).
- `teams[].ppda_opp_passes_per_def_action` - the other team's passes / our pressures, whole pitch.

On Metrica's 2 pro games, 4 halves, 8 team-halves (`tools/n0_24d_ppda.py` -> `metrica_ppda.json`; summary
`tools/n0_24d_summary.py` -> `summary.json`). Ours = the pipeline on Metrica's positions (passes on the spell state, like
`run.py`), perfect ball and our two ball-noise models (2 seeds). Two truths from Metrica's events + tracking:

- **same rule**: our definition with perfect inputs (Metrica's sequences, carrier from the events, 2 m from tracking).
  Tells us what our ball noise does to the number.
- **standard PPDA** (the usual stat): the other team's passes tried in its own 60% / our tackles, ground challenges,
  interceptions, steals and fouls in that zone. Tells us whether our number says what a coach expects it to say.
  (Metrica tags a lot of loose-ball pickups 'interception'; without them the 'strict' figure is ~1.4x higher. Both are
  below the 7-15 quoted for Opta data because Metrica labels more defensive actions.)

## Result

**Pressures: the count holds.** Ours / same-rule truth per team-half (min / median / max):

| ball | pressures | other team's passes | which team pressed more (4 halves) |
|---|---|---|---|
| perfect | 0.77 / 0.92 / 1.04 | 0.93 / 1.01 / 1.08 | 3 / 4 |
| streak noise | 0.51 / 0.76 / 0.88 | 0.86 / 0.96 / 1.05 | 2 / 4 |
| flicker noise (like our clips) | 0.90 / 1.07 / 1.18 | 1.02 / 1.11 / 1.20 | 3 / 4 |

Truth is 73-129 pressures per team per half (~150-260 per 90 min), so S7's 288-306 per 90 min on our clips is in the
right range for a match with more duels.

**PPDA: right size, wrong story.**

| ball | ours | same-rule truth | standard truth | rank vs standard (8) | which team presses more vs standard (4) |
|---|---|---|---|---|---|
| perfect | 2.4 / 3.0 / 3.5 | 2.0 / 2.7 / 3.2 | 2.6 / 3.5 / 4.3 | 0.64 | **1 / 4** |
| streak | 3.0 / 3.5 / 3.8 | | | 0.24 | **0 / 4** |
| flicker | 2.2 / 2.8 / 3.2 | | | 0.79 | 3 / 4 |

- The number lands near the truth size, but inside a match it does not tell which team pressed more: 1 of 4 halves
  even with a perfect ball. Even our own rule with perfect inputs agrees with the standard PPDA only 2 of 4
  (one a tie; rank 0.50). A player 2 m from the carrier is not a tackle or an interception.
- The 3 of 4 under flicker is most likely chance: the perfect ball gets 1 of 4 and the streak noise 0 of 4.

## Tried, not switched on

Lab `--zone`: our PPDA with the standard zone (other team's passes from its own 60%, pressures on a carrier there):
which team presses more 1 / 4 with every ball, rank 0.17-0.40. Worse. **No pipeline change.**

## What it means

- `ppda_opp_passes_per_def_action` stays HIDE. The usual PPDA needs tackles and interceptions, which we do not detect.
- `pressures_applied` (team) is now checked: about right with any of our ball models (0.76-1.07x). Showing it as Beta
  is Daniel's call (under Waiting for Daniel). Per player it stays hidden until M2 (one track per player).

Tests: `tests/test_n0_24d_ppda.py` (6); full suite 606 pass. $0, nothing on Modal.
