# S2: set pieces from player motion? No - restarts stay on the ball rule (9 Oct, worker, local + free runner, $0)

**Question:** switch restarts (throw-ins, corners, goal kicks, free kicks) from today's ball rule (ball 'dead' >= 3 s)
to the player-motion rule (everybody slow for >= 3-4 s), or to both together? The 3 Oct test on Metrica and on the
5-min clip could not decide; this test uses the whole SFK-BP first half as the app exported it (frames fetched 8 Oct).

## 1. Against Veo's own event list (29 covered minutes) - can't tell them apart
`tools/s2full.py` -> `s2full.json`. Veo lists 32 restarts in minutes 11-31 + 44-45 (6 more in 46-51).

| rule | ours | Veo restarts with one of ours in the same minute | ours with no Veo restart |
|---|---|---|---|
| ball (app today) | 34 | 23/32 | 11 |
| motion 1.4 m/s, 3 s | 34 | 23/32 | 11 |
| motion 1.0 m/s, 3 s | 14 | 11/32 | 3 |
| union (ball + motion 1.4/3) | 57 | 29/32 | 28 |

But the same number of RANDOM moments scores almost as well (34 random: 21/32, 57 random: 27/32) - Veo only gives the
minute, so this score is close to useless. Hence a blind check by eye.

## 2. Blind check by eye (the real answer)
46 moments from the covered minutes, shuffled, 4 full frames each (t-4, t-1.5, t, t+2 s), nothing drawn, graded before
opening the key (`blind_grades.json`, key `blind_key.json`, sheets `results/qa/s2/*.jpg`).

| found by | sheets | restart seen | can't tell | open play |
|---|---|---|---|---|
| both rules | 14 | 7 | 3 | 4 |
| ball rule only | 16 | 6 | 2 | 8 |
| motion rule only | 16 | **1** | 7 | 8 |

- The motion rule's own finds are almost never a restart (1/16): they are slow build-up, players walking back after an
  attack, waiting near a box. Adding them (the union) brings ~28 extra moments for ~2 real restarts.
- The ball rule is right about 4 times in 10 (≈19 of its 46 in these minutes), when both agree it is 1 in 2.
- So: **nothing switched; restarts stay on the ball rule.** The motion rule stays in the code as an option.

## Notes
- The ball rule itself still over-counts (about 6 in 10 are open play by eye) - restart counts stay as they are labelled
  in the app now. A better fix needs the ball (or the ball leaving the pitch), not player speed.
- The follow-cam sees ~13-16 players, so a 'median speed of everyone' is noisy; on pro data with all 22 it works better.
- Veo lists set pieces in minutes 1-9, but kick-off is at 9:15 in the video (periods/SFKBP1109.json); our export starts
  there, so those minutes were not used.

Files: `tools/s2full.py`, `tools/s2_strips.py`, `tests/test_s2full.py`.
