# N0-24e (24 Oct, worker, local $0, nothing on Modal): restarts on pro data

Card: 24 Oct card says nothing worse than 23 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

The app's restarts (`stats.restarts[]`: throw-ins, corners, goal kicks on the timeline, HIDE) had never been checked
against a truth. On Metrica's 2 pro games (4 halves, 194 restarts: 115 after the ball went out = throw-ins, corners,
goal kicks; the rest free kicks, kick-offs, 1 penalty). `tools/n0_24e_restarts.py` -> `metrica_restarts.json`, summary
`tools/n0_24e_summary.py` -> `summary.json`. Ours = the pipeline on Metrica's positions (metricalab.ours: perfect ball,
streak and flicker noise, 2 seeds). Found = one of ours within 5 s.

## Result

**Old rule: blind.** It found 2 of 194 with a perfect ball (4 flicker, 19 streak). Metrica blanks the ball while it is
out of play, and the old possession model, with no ball to look at, just keeps the last state, so a stoppage was never
'dead'. The same happens to us whenever the ball finder loses a ball that went out.

**New rule (on by default, `IPANEMA_GAP_DEAD_S=0` = old):** a gap of 2 s or more with no ball, whose last sighting before
it or first one after it is within 3 m of a line (or off the pitch), is a stoppage. The restart spot is where the ball
comes back (first 0.2 s), the team is the player nearest to it then, and a goal kick is a spot within 6 m of the end
line and 10 m of the centre (was 3 m: real goal kicks are taken from the 5.5 m line).

| ball | out-of-play restarts found | all found | ours that are real | kind right | team right | ours / truth per 90 |
|---|---|---|---|---|---|---|
| perfect, old | 2 / 115 | 2 / 194 | 29% | 0% | 50% | 3 / 92 |
| perfect, new | **98 / 115** | 112 / 194 | **90%** | **81%** | **97%** | 59 / 92 |
| flicker, new | 96 / 115 | 109 / 194 | 89% | 69% | 85% | 58 / 92 |
| streak, new | 98 / 115 | 112 / 194 | 81% | 61% | 51% | 65 / 92 |

Free kicks are still almost never found (5 of 64): the ball stays in view and a player stands by it, and we never see
the foul. Kick-offs 9 of 14.

Variants (same table in `summary.json`): 3 s gap = the same; only the sighting before the gap = slightly fewer;
6 m from the line = 6 more found but more extras with streak noise. Kept 2 s / 3 m / either side.

## On our footage: no change

Our ball picker always leaves a ball (gaps up to 1 s are bridged), so the rule does not fire on any of our saved inputs:
the 2 app clips and 5 more 2.5-min windows (SFK-BP, AIK, Vallentuna x2) give exactly the same restarts, kinds and teams
(`tools/n0_24e_clips.py` -> `clips.json`), and in the whole Vallentuna export there is 1 ball gap of 2 s or more
(half-time, not near a line). So there were no new moments to look at by eye, and the app's restarts stay as they are
(S2: right about 4 in 10). What changes for the app: a goal kick from 3-6 m out is now called a goal kick, and a ball
that goes out and is lost for 2 s+ will now make a restart.

What this tells us: the restart logic itself (when, where, which kind, which team) is right on pro data once the
stoppage is seen; our problem on Veo is seeing the stoppage (the picker keeps picking something while the ball is out).

Also: `tools/metricalab.py` (nightly) now passes the players to `restarts()` like `run.py`, so the nightly's
Metrica set-piece numbers will jump (found 1/51 -> ~38/55 per half) - not a regression.

Tests: `tests/test_n0_24e_gap_restarts.py` (7); full suite 605 pass, 3 fail only because this machine lacks the `supervision` package (same without this change). $0, nothing on Modal.
