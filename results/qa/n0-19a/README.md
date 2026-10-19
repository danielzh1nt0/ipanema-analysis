# N0 19 Oct 00:20 UTC (worker): new-venue camera base (ipanema/venue.py) tested

Nightly card: the 18 Oct card says "nothing worse" than 17 Oct. The 19 Oct card is not due yet (before 03:00 UTC).
No new main-session code since 20:20, so another piece without tests: `ipanema/venue.py` - the code behind Modal's
`venue_base` (a new ground: sample frames, paint lines, find where the Veo mast stands from the lines alone).

## Faults fixed
- **No frame placed -> crash.** If no sampled frame could be placed under the reference (Edsberg) base, or there were no
  frames with enough painted lines, `solve_base` died with a TypeError *after* the line model had already run. Now it
  hands back the reference base with `"error"` in the report; `venue_base`'s held-out test then says REJECTED and keeps
  the old base, as it does for any base that doesn't win.
- Same for the second step (no frame placed under the best base: was a division by zero).
- `strip` with no frames crashed (np.vstack of nothing); now one black row.

## Is the method right? (local, 16 min CPU, `tools/n0_19a_venue_synth.py`)
Line masks drawn from the KNOWN SFK-BP camera (8 frames), search started from a base 3.3 m off:
- solved base 0.71 m from the truth (x 53.19 vs 53.33, y 66.54 vs 67.23, height 4.66 vs 4.73 m);
- held-out check accepts it clearly (cost 0.97 vs 2.91 for the wrong start base);
- by eye (`venue_synth_truth_vs_solved.jpg`, truth white, solved in colour): far touchline, boxes, centre circle and
  halfway line sit on the truth in all 8; only the near ends of the goal lines (bottom of #6, #7) drift tens of px apart.
- Not perfect even on perfect masks: the coarse search picked y 65.5 over the nearer 67.0 (it only refines the poses
  fitted under the start base), and the refine stops after 50 steps with cost 1.10 while the true base scores ~0.86.
  Nothing changed for that - it would change Modal results; worth a look if a new ground is ever solved.

## Tests
`tests/test_venue.py` (7): grid, true base beats a wrong one on real rendered lines, held-out picks the true base,
search + refine finds a known base (fake fitter), nothing-placed and no-masks cases (2 fail on the old code), strip.
Local suite 435 pass, 6 skipped, 3 need supervision (not installed here). $0, nothing on Modal started.
