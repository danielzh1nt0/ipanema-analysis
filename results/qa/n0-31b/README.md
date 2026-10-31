# N0-31b (31 Oct 2026, worker, $0, nothing on Modal)

**Nightly:** no 31 Oct card by 04:20 UTC, so the backup nightly was pushed (triggers/nightly.txt, free runner, 04:21-04:43).
The 31 Oct card says **nothing worse than 30 Oct**: tests 198/198 pass (the 30 Oct failure is fixed), all 7 player checks ran,
every number is the same as on 30 Oct. Nothing goes to the top of the queue. M2 / PC2 skipped (main session / after M2).

**Small extra check - shot spots vs attack direction** (tools/n0_31b_shotspots.py, tests/test_n0_31b_shotspots.py, 5 tests).
Every Veo shot with a clicked spot must be on the pitch, in the half the shooting team attacks, under 40 m out,
and its `distance_m` must match the distance to that goal's centre. A wrong attack direction (like AIK's, N0-26b)
would show up as shots in a team's own half.

- Vallentuna: 16 shots, 11 with a spot - all 11 pass (8.9-24.7 m). By eye on `vallentuna_shotmap.png`: every spot
  is in or around the box the team attacks; two light-team shots are from tight angles near the box corner (plausible).
- SFK-BP (7) and AIK (16): no spots yet (V1c, Daniel's clicks), so nothing to check; the test will catch a wrong
  direction once they are clicked.
- Side note, not changed: `located_by` says 'end only, team by eye' even when the spot was clicked; nothing in the app reads it.

Files: shotspots.json, vallentuna_shotmap.png.
