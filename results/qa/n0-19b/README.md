# N0-19b (19 Oct, worker, local $0): clips now get the attack direction Daniel fixed by eye

**What was wrong.** Which way each team attacks drives field tilt, high turnovers, forward passes, attacking third and
the shot end. Our automatic guess is close to a coin flip (confidence 0.01-0.10 on our clips), so on 4 Oct Daniel's
by-eye answer went into `periods/<match>.json` ("attack_right_A"). But a 5-min clip cut from a match has its own id
(`SFKBP1109_s1200`, `p15u-vs-aik-2026-09-21-bd09_s2520`), and `run.periods_direction` only looked for a file with
exactly that id - so **every clip ignored the by-eye answer** and used the guess.

**Effect in the app.** The AIK clip run (results/modal/p15u-vs-aik-2026-09-21-bd09_s2520.txt, log of 17 Oct):
ball drift confidence 0.03 / 0.01 -> keepers rule -> "A <- B ->" = **the wrong way** (periods file: dark SFK attacks
right in the first half; the clip is at 42-47 min, inside the first half 6:06-53:45). By eye on the clip's own debug
picture f004496: goal kick from the LEFT goal with the dark team spreading out -> dark defends left, attacks right,
agrees with the file. So the AIK clip's direction numbers in the app are mirrored (field tilt, high turnovers,
forward passes, attacking third). The SFK-BP clip was right by luck (keepers rule gave A ->, same as the file).
The full matches were never affected (their ids match their files).

The median-x fallback is no help here either: on the AIK clip's players it reads A 58 m vs B 57 m (a coin flip, and
the wrong way); on SFK-BP 36 vs 43 (right).

**Fix** (`ipanema/run.py` periods_direction): a clip id `<match>_s<start>[_d<dur>]` reads `<match>.json`; the clip's
middle (start + dur/2, dur 300 s if not given) picks the nearest period; clips are not mirrored, so in the 2nd (4th, ...)
period the value is flipped. No `periods_s` in the file -> None (can't tell the half) -> automatic guess as before.

| id | before | now |
|---|---|---|
| SFKBP1109_s1200 | None (guess: A ->) | A -> (file) |
| p15u-vs-aik-2026-09-21-bd09_s2520 | None (guess: **A <-**) | A -> (file) |
| SFKBP1109_s4200 (2nd half) | None | A <- (flipped) |
| reymersholm _s1500_d300 | None | None (no file) |
| full matches | from file | unchanged |

**Tests.** tests/test_periods_direction.py (+3: clip ids, period from the clip's middle incl. half-time, file without
periods -> None; 2 fail on the old code). New tests/test_direction_auto.py (8) for the untested automatic chain
(ball drift + confidence, jumps ignored, keepers vote, < 10 keeper rows -> None, median-x fallback leaves keepers out,
always opposite teams) - no faults there.

**Reaches the app** at the next AIK clip run (Modal - Daniel's go); nothing re-run. Committed with [skip ci] so the
push does not start the Modal deploy workflow.
