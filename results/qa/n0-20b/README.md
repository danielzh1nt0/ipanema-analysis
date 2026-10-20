# N0-20b (20 Oct 08:20 UTC, worker, local + free runner, $0, nothing on Modal)

**Nightly card 20 Oct:** worse than 19 Oct = nothing (tests 140/140, ball 22/34). No new main-session code since the 04:20 run, so I took untested code the app shows: `metrics.field_tilt_and_entries` (field tilt % and final-third entries, Shooting tab, "Where did they get in?").

## What was wrong
Every crossing of the 2/3 line while a team "had" the ball counted as an entry. On the app's SFK-BP clip (replayed from the app's own picker inputs, `tools/entrylab.py`, numbers match the app exactly: dark 20 / light 10 entries in 5 min) **11 of dark's 20 come less than 2 s after the previous one** - e.g. 3 entries at 8.01, 8.11, 8.17 s.

## By eye (free runner sheets, `results/free/n0-20b`, ball pick ringed red, 2/3 line yellow)
Most of these are not play at all: the ball pick parks on a **spot on the far touchline that sits right on the 2/3 line** (by the ad board) and every wobble counts. See `grades.json`.

| stretch | old rule | new rule | real |
|---|---|---|---|
| 8.0-10.7 s (stoppage) | 6 | 1 | 0 |
| 13.3-15.5 s | 4 | 3 | 0 |
| 132.1-132.3 s | 3 | 1 | 0-1 |
| 208.8-209.0 s | 2 | 1 | can't tell |
| 232.1-232.2 s | 2 | 1 | 0 |
| **total** | **17** | **7** | **0-2** |

## Fix (on by default, reaches the app at the next run)
Re-arm rule: after an entry, the next one counts only once the ball has been 3 m back behind the line, or the team out of control of it for 1 s. Clip: dark 20 -> 11, light 10 -> 9. Field tilt unchanged (12 / 44). `IPANEMA_ENTRY_REARM=0` = old rule. Tests: `tests/test_final_third_entries.py` (7), `tests/test_n0_20b_frames.py` (1).

## Not fixed (bigger, not this item)
- The touchline spot on the line is the same clutter spot S8 found; it still makes fake entries when the pick teleports there (a teleport re-arms the rule, m14).
- Entries and tilt read the raw (flickery) possession state, not the de-flickered one the counted stats use. Tried: spell state changes *which* team gets entries (dark 7, light 16) - not graded, not switched.
