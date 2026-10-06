# S8 closed (6 Oct, worker, local, $0)

Question: the possession state flipped 27-33 times a minute (real play ~2-3), which blew up passes, sequences, turnovers and pressing.

Re-checked today with the current code on the exact app inputs of both 5-min clips (`PYTHONPATH=. python tools/s8close.py`, numbers in `s8close.json`):

| per 5 min | SFK-BP raw | SFK-BP counted | AIK raw | AIK counted |
|---|---|---|---|---|
| flips per minute | 27.2 | **3.2** | 33.2 | **2.0** |
| sequences | 81 | 32 | 93 | 24 |
| turnovers | 20 | 21 | 13 | 17 |
| passes | 87 | 59 | 107 | 79 |
| graded passes kept (real /20, fake /8) | 20 / 8 | 14 / 2 | - | - |

- "Raw" = the state shown as possession % and the ball display (unchanged, 82/99 who-has-the-ball).
- "Counted" = the spell state (take 1.5 s / join 3 s) that run.py uses for sequences, turnovers and passes since 2-4 Oct.
- All three S8 lines are done: (a) finder retrained on balls at feet - mixed, and worse at night in the B6 blind check, not used; (b) heavier WASB near feet - no gain; (c) spell state - in the app's code.
- What is left is not flicker: the spell filter also drops 6 of 20 real passes (quick one-touch / intercepted). That is S9 (after the demo).
- New test `tests/test_s8_close.py` keeps the counted state at <= 4 flips a minute and fake passes kept at <= 2 on the real clip.

Side fix: `ipanema/panorama.py` used `np.cross` on 2-D points, which numpy 2.5 no longer allows (test_panorama_extend failed locally); replaced by the same formula written out. Full suite here: 292 passed, 2 skipped.
