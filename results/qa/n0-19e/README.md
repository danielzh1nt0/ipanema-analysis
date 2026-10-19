# N0-19e (19 Oct 16:20 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card 19 Oct:** worse than 18 Oct = nothing (tests 134/134, ball 22/34, passes 15% / possession 8 pts off on pro data). No new main-session code since 12:20, so I ran the test suite with a coverage count and picked the least-tested code that every app run uses: `ball.fuse_candidates` (joins the RF-DETR ball finder's guesses with WASB's into one list per frame; no test at all).

## What was wrong (small)
- A WASB guess was added to the **first** finder guess within 12 px in the list, not the nearest one.
- One finder guess could take the "both agree" bonus several times (two WASB guesses on one ball).
- On the SFK-BP clip with the app's files this touches 368 of 53,952 WASB guesses (frames changed: ~4%).

## Fix, as an option only
`IPANEMA_FUSE_NEAREST=1` (default off = today's rule): nearest finder guess, bonus once (strongest WASB guess), WASB guesses never joined to each other. `tests/test_fuse_candidates.py` (7, incl. the old behaviour pinned and "default = old").

## Does it score better? No
`tools/fuse_check.py` -> `fuse_check.json`, same-job finder files, through the app's picker:

| finder | 34 checked balls | B4 (285) | who has the ball (99) | graded passes real / fake kept |
|---|---|---|---|---|
| app finder (B7), old rule | 29 | 284 | 72 | 15 / 4 |
| app finder (B7), new rule | 29 | 284 | 72 | 15 / 4 (pass count 81 -> 84) |
| other 4 finders | +-1 each way | +-1 | +-1 | +-2 |

Picked ball differs on 64 of 8,992 frames (13 short spells). Blind A/B pictures of 34 of them (free runner, `results/free/n0-19e`, side A/B random, answers written before opening `ab_key.json`): **old rule right 7, new right 5**, 8 neither on the ball, 14 can't tell. Not better -> stays off.
