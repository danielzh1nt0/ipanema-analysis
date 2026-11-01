# N0-01b (1 Nov, worker, free runner + local, $0, nothing on Modal): do the app's numbers add up? + passes on the wrong player card

No 1 Nov card by 04:20 UTC -> backup nightly pushed; the 1 Nov card says nothing worse than 31 Oct (tests 203/203).
M2 (main session) and PC2 (waits for M2) skipped.

## 1. Consistency sweep (new)
`tools/n0_01b_consistency.py` checks ~30 rules that must hold whatever the video shows, on the 4 app copies:
team totals vs the passes / sequences lists, completion %, forward share, progressive, passes per sequence, team distance /
pressures = sum of players, possession A + B = 100 and = seconds, balls lost by A = balls won by B, per-player
forward + sideways + backward = passes, completed <= passes, thirds add to 100, high-intensity <= distance, time on ball <=
time seen, etc.

Almost everything adds up. Found:
- **Player cards vs team totals** (SFK-BP, AIK, Vallentuna): a team's players had more passes than the team (SFK-BP B 78 vs 75,
  Vallentuna B 260 vs 256) and 'a better pass was open' didn't sum (e.g. Vallentuna A 117 team vs 110 players). Cause below.
- **The SFK-BP clip copy (SFKBP1109_s1200)**: its stats.json and match_data.json come from different runs (only 38 of
  273 player ids are shared; team sequences 112/102 vs 81 in the list; 13 restarts vs 31). Not fixed here: needs a fresh
  export of the clip (a run, Waiting for Daniel). The full SFK-BP copy is fine.

## 2. Passes credited to a player listed in the other team
A pass's team is the shirt read at the touch; a player's team on his card is his track's team over the whole match.
They disagree on **7-13% of passes** (SFK-BP 12/167, clip 11/87, AIK 20/225, Vallentuna 56/487), and the pass then sits on
the card of a player of the other team.

**Blind by eye** (free runner cut 41 sheets: passer ringed 'P', one player of each team as '1'/'2' in random order;
graded before opening the key, `grades_blind.json`; colours from the reference players: SFK-BP A black / B white,
Vallentuna A black / B red, 0 conflicts):

| | passer wears the pass team's shirt | the card's team | can't tell | not a player |
|---|---|---|---|---|
| 30 disagreeing passes | **20** | 6 | 3 | 1 (orange bib) |
| 11 agreeing controls | 6 of 6 decided right | - | 4 | 1 |

So about 3 in 4 of those credits are another player's pass (the track slid onto him: M2's ID problem).

**Change** (`ipanema/analytics.py`, `IPANEMA_PASS_OWN_TEAM`, default on; `=0` = old): a pass counts on a player's card
(passes, completed, forward/sideways/backward, progressive, received, quality, better option) only when it is his team's
pass. Team totals unchanged. On the 4 copies (`tools/n0_01b_rebuild.py`, old rule reproduces the stored numbers exactly
first): 99 passes leave player cards (~20 wrong credits gone for every ~6 right ones lost), 137 player rows change, and the
consistency sweep now finds 0 broken rules on SFK-BP, AIK and Vallentuna (clip: only the different-runs fault).
Pro data (Metrica) unaffected: ids and teams are true there, so the two always agree.

Side note: 4-8% of passes go from a player to himself (SFK-BP 8/167, Vallentuna 37/487) - a knock-on or a slide; not changed.

## Files
`tools/n0_01b_consistency.py`, `n0_01b_moments.py`, `n0_01b_frames.py` (free runner), `n0_01b_score.py`, `n0_01b_rebuild.py`;
`tests/test_n0_01b_consistency.py` (9). Sheets in results/free/n0_01b. Suite: 853 pass, 3 need supervision (not installed here).
Reaches the app at the next run.
