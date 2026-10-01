# M1b: does the far side read faster? (1 Oct, free, local)

Exact app inputs of SFK-BP and AIK, speed layer old (1 s average) vs new (noise-aware smoother). `tools/motionlab_m1b.py`,
numbers in `motionlab_m1b.json`, chart `speed_by_scale.png`.

**Answer: no, not on the whole clip.** Grouped by how coarse the calibration is at the player's feet (metres per pixel up/down
the image: ~0.05 near the camera, 0.3-0.9 at the far touchline or when the camera zooms out), the far/coarse groups read
*slower*, not faster (SFK-BP median 4.6 km/h far vs 6.5-7.1 near; AIK 2.9-4.7 vs 4.9-5.5). The "far 9.7 vs near 5.5"
came from the 28 hand-picked moments, which were chosen among >25 km/h readings, i.e. noise spikes, and those happen
where 1 pixel = 0.3-0.9 m. The calibration itself looks sane (far touchline at the right place, scale grows smoothly).

**Fix for the spikes:** the speed layer now uses the calibration to know how noisy each foot point is in metres
(2 px times the local metres-per-pixel) and a Kalman smoother trusts far, coarse points less. Speed = smoothed velocity.

| SFK-BP | old | new |
|---|---|---|
| by-eye moments in range (hidden counts OK) | 14/21 | 16/21 |
| km/h outside the by-eye range, summed | 44.8 | 15.4 |
| metres per player-minute | 96 | 102 |
| median / p95 km/h | 5.5 / 16.3 | 5.9 / 17.4 |
AIK: 77 -> 85 m per player-minute, median 4.7 -> 5.1 km/h (no by-eye key there).

Fixed: far jogger 21 -> 12 km/h (#15), standing player 12 -> 6 (#7), far walker 24 -> 17 (#18). New miss: a far walker that
was hidden now reads 20 (#4). Still wrong: two walkers near the middle read 10-11 km/h (#1, #2; clean tracks moving
~3 m/s - either the by-eye grade from 2 stills is off or the pitch size 106x64 is too big; needs SFK-BP's real size).

Honest limits: the one setting (noise 2 px vs 1 m/s² change) was picked from 5 on the same 21 moments (14-16/21 for all);
21 moments is a small key; ID swaps are untouched (M2). The layer stays OFF in the app (IPANEMA_MOTION=1); run.py now
passes the calibration so the next run uses the new smoother. Test: tests/test_motion.py (far standing player reads
still, far runner keeps 18 km/h where the 1 s average reads ~11).
