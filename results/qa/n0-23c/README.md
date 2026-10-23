# N0-23c (23 Oct, worker, local $0): forward pass share + progressive passes, and a hidden test fault

Card: 23 Oct nightly says nothing worse than 22 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## 1. Pass direction figures in the app (never checked before)

`stats.json teams[].forward_pass_share_pct`, `teams[].progressive_passes` and per player `progressive_passes` come
from `analytics.passes`: gain = receiver's spot at his first touch minus passer's spot at his last touch, towards the
attacking goal. Forward > 3 m, backward < -3 m; progressive = gain >= 10 m and the pass reached a team-mate.

Checked on Metrica's 2 pro games, 4 halves (`tools/n0_23c_progressive.py` -> `metrica_progressive.json`).
Truth = Metrica's labelled passes (ball start/end). Ours = the pipeline's passes on Metrica's positions, with a perfect
ball and with our two ball-noise models (2 seeds each).

| | perfect ball | 'streak' noise | 'flicker' noise |
|---|---|---|---|
| each matched pass: forward / sideways / backward right | 94% | 89% | 90% |
| truth progressive passes we also call progressive | 84% | 75% | 75% |
| our progressive passes that are truly progressive | 97% | 87% | 95% |
| team progressive count vs truth (min / mean / max) | 0.71 / 0.92 / 1.19 | 0.57 / 0.85 / 1.09 | 0.70 / 0.90 / 1.19 |
| team forward share vs truth, points (min / mean / max) | -7.8 / -4.2 / -1.2 | -7.4 / -1.7 / +3.2 | -8.0 / -1.6 / +6.8 |

So: **the figures hold.** Progressive passes read a little low (about 1 in 10 under), forward share a few points low.
Most misses are borderline (Metrica's x has 1.05 m steps; many missed ones are 10.5 m in truth and 9-10 m ours) or a
pass chain cut in two. Our pass COUNT is still 1.1-2.1x the truth (known, P-PASS); the share and the progressive
count are much less hurt by that than the pass count itself.

Tried (`gain_variants.json`): measuring the gain from the ball's spot at the two touches instead of the players'
spots, and the mean of both. Both are worse on every row (perfect ball: kinds right 81% / 90% vs 94%, progressive
found 50% / 65% vs 85%) - the ball is already ~2 m on its way at the passer's last touch and ~2 m short of the
receiver at his first. **Today's rule stays; no pipeline change.**

App copies today: SFK-BP clip forward 39 / 42%, progressive 8 / 3; AIK clip 44 / 34%, 14 / 16 (AIK clip's attack
direction is mirrored in the app until its re-join, N0-19b, so its forward share is the wrong way round there).

## 2. Hidden test fault (fixed)

Two tests (`test_track_loop.py`, `test_watchdog.py`) put a fake `ultralytics` module into `sys.modules` and never took
it out. With the real ultralytics + torch installed (any GPU box) the next file, `test_tracking_batch.py`, then failed
2 tests; the nightly runner has no torch, so that file returns early and nobody saw it. Reproduced (2 failed), fixed
with `monkeypatch.setitem` (removed after each test). Full suite with torch installed: 581 pass, 0 fail.

Tests: `tests/test_n0_23c_progressive.py` (5): forward / sideways / backward / lost pass with the team attacking left,
the 10 m line, the Metrica truth direction, pair matching, and a check that no test leaves a fake module behind.
