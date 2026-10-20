# N0-20e (20 Oct 20:20 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card 20 Oct:** worse than 19 Oct = nothing. No new main-session code since 16:20.

**Coverage run** (full suite, 495 pass): the one app step with an untested body was `possession.restarts()` - the set pieces on the app timeline ("throw-in · A", corner, goal kick) and the kick-off check that Veo goals use. Now 6 tests (`tests/test_restarts.py`).

## What I found
The restart spot was the median ball position from the stoppage's last second **up to 1.5 s after it ended**. On our clips the ball is seen on every one of those frames, so 45 of the 75 frames come after the stoppage, when the ball is often already on its way. The docstring (S5) says "the last second".

## Tried: spot from the stoppage's own last second (`spot="last1"`)
Replay on the two app clips' exact inputs (`tools/n0_20e_restarts.py`, `replay.json`): SFK-BP 7 restarts, the kind changes on 2 and the spot moves a median 6.4 m; AIK 1 restart, same kind.

Graded **blind by eye** on zoomed free-runner sheets (`results/free/n0-20e/zoom/*.jpg`: every 0.5 s from 4 s before to 3 s after, both spots drawn as unlabelled crosses 1/2, key opened after grading):

| moment | what the video shows | better spot | kind old / new |
|---|---|---|---|
| sfk00 6.5 s | throw-in, taken ~1.5 s after the stoppage ends | **new** (old 19 m off, where the ball went) | throw-in / throw-in |
| sfk01 161 s | can't tell (far corner) | - | throw-in / corner (Veo lists throw-ins here) |
| sfk02 179 s | throw-in | both within ~2 m | **throw-in** / stoppage (y 3.4 m, just past the 3 m rule) |
| sfk03 196 s | no clear restart | - | throw-in / throw-in |
| sfk04 241 s | throw-in near the corner | new (by ~4 m) | throw-in / throw-in |
| sfk05 261 s | goal kick, keeper plays it ~1.5 s after the stoppage ends | **old** (new = in the net, where the ball lay) | goal kick / goal kick |
| sfk06 280 s | play on (header duel) - not a restart | - | stoppage / stoppage |
| aik00 177 s | play on (scramble in the box) - not a restart | - | stoppage / stoppage |

Spots: new better 2, old better 1, 1 tie; kinds: new loses a right throw-in and turns another into a doubtful corner. **No clear gain, so the default stays as it was** (`IPANEMA_RESTART_SPOT=after`). `last1` is an option only.

Why neither window is right: the stoppage's "end" is not the restart. In sfk00 and sfk05 the throw / goal kick comes ~1.5 s after the dead state ends; during the stoppage the ball is in the net or off the pitch. A good spot needs the moment the ball is played (still ball -> moving), not a fixed window.

Also seen (not new, S2): 2-3 of 8 'restarts' are not restarts at all - the ball pick on a spot off the pitch makes the play look dead.

Files: `tools/n0_20e_restarts.py` (replay + blind moments), `tools/n0_20e_frames.py` (sheets, `--zoom 800`), `moments.json`, `moments_key.json`, `replay.json`. Tests: `tests/test_restarts.py` (6), `tests/test_n0_20e_frames.py` (3, dry run on stand-in clips); suite 495 + 9 new (all 129 tests in the possession/restart files re-run after the change: pass).
