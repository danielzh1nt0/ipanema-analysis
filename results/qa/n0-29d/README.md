# N0-29d (29 Oct, worker, local $0, nothing on Modal): the runs' 'with the ball' label

Card: 29 Oct card says nothing worse than 28 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked (never before)
Every sprint in the app's Runs layer (`stats.metrics.runs[]`) carries `with_ball`: did the runner's TEAM have the
ball when the run started (True = run in possession, False = chasing / pressing / recovering, None = nobody). N0-25a
checked that the runs are real; nobody checked this label.

The app read it from the **raw** possession state at the run's first frame. That state flickers, and a loose frame
gives None.

## How (no clicking)
Metrica's 2 pro games, 4 halves. Runs = the app's own `metrics.runs` on Metrica's positions as the pipeline sees them
(1,875 runs), so only the label is graded. Truth = the team of Metrica's possession chain (on-ball events) covering
the run's first frame (90% of runs; the rest start in a contested / dead moment and are not scored).
Ball: perfect, and our 'flicker' noise (ball dot hopping to a nearby opponent, like our clips; 2 seeds). The 'streak'
noise drops players at random every frame, so no run lasts 0.8 s there - not usable for this check.
`tools/n0_29d_runball.py` -> `metrica_runball.json`, picture `runball.png`.

| label read from | runs labelled | labels right | share of sprints in possession, off |
|---|---|---|---|
| raw state, first frame (**app until today**) | 61% / 66% | 96% / **77%** | 4.5 / 6.6 pts |
| **spell state, first frame (new default)** | **83% / 88%** | **98% / 93%** | **2.3 / 2.4 pts** |
| spell state + look back 3 s (lab option) | 95% / 97% | 95% / 90% | 1.9 / 2.3 pts |
| spell state, most of the run (lab only) | 98% / 99% | 92% / 88% | 2.3 / 2.6 pts |
| (baselines: 'always out of possession' right 54%; every run gets the team's possession share: 9.6 pts off) | | | |

(perfect ball / flicker ball.) With our kind of ball the app labelled only half of all runs right (51%: a third got
no label, a quarter of the rest were wrong). The spell state is better on every row of both ball models, so it is
the new default. The 3-s look-back labels more runs but is 3 points less often right -> option only.

## On our two app clips (exact picker inputs, `tools/n0_29d_clips.py` -> `clips.json`)
Same 37 runs as N0-25a. Without a label: SFK-BP 8 of 26 -> 1, AIK 1 of 11 -> 1. Labels flipped: 1 per clip.
Against the SFK-BP by-eye who-has-the-ball key (58 moments, nearest within 2 s): of the newly labelled runs 4 of 5
agree with the key; the 2 flipped runs have no key moment near. The old labels that disagree with the key (138.5 s,
138.7 s, 215.7 s: A 'with the ball' where the key says B) stay wrong - the state itself is wrong there.

## Change
`metrics.compute` gives `metrics.runs` the spell state (`cstate`, already passed by run.py) for the label.
`IPANEMA_RUN_BALL=raw` = old, `=spell_back` = with the 3-s look-back. Runs themselves (which, where, how fast) are
unchanged. Reaches the app at the next export (no Modal needed for the code; the numbers change when a match is
re-exported).

Tests: `tests/test_n0_29d_runball.py` (6): first frame vs look-back, compute reads the spell state by default (raw
when none is given), the old switch, lab labels = app labels, truth / scores, dry run. Full suite 776 pass, 4 skipped.
