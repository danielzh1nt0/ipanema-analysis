# N0 18 Oct 12:20 - shots, sprints and pressure points get tests

Morning card (results/nightly/2026-10-18.md): nothing worse than 17 Oct. No new main-session code since 08:20, so the
worker tested the part of `ipanema/metrics.py` every app run uses but no test touched (coverage 72%): shots, high
turnovers, sprints (`runs`), pressure points, the momentum strip (`tilt_windows`). tests/test_metrics_shots_runs.py, 10 tests.

Found and fixed:
- **Sprints cut off at a track gap were thrown away.** When a player's track had a gap (> 3 frames), the sprint up to the
  gap was dropped instead of kept. Now kept. On the exported frames of the repo copies (10 fps, rebuilt by
  `tools/n0_runs_gap.py`): SFK-BP 51 -> 53 sprints, Vallentuna 735 -> 776 (+6%), AIK piece 5 -> 5; every old sprint is
  still there; the new ones are typical (median 0.8-1.0 s, 6.4-6.9 m/s). The test fails on the old code. runs_gap.json.
- Shot outcome had two labels for the same thing ("blocked or saved" and "saved or blocked"); now one ("saved or blocked").
  Nothing in the app keys on the text.

Found, NOT changed (on purpose):
- Our own shot detector can never call a goal: its rule wants the next restart at the centre spot marked 'unknown',
  but restarts there are marked 'stoppage'. Checked on the three full matches' exports: no centre restart follows any
  of our detected shots (SFK-BP 8 shots, AIK 4, Vallentuna 16), so a fix would change nothing today, and the app's
  goals come from Veo's list anyway. Written into the code and pinned by a test, so a later change is deliberate.

Suite: 409 pass, 5 skipped, 4 need torch/supervision (not in this sandbox). Reaches the app at the next paid run.
$0, nothing on Modal, no free-runner push (pure local code + test).
