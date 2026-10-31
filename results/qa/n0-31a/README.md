# N0-31a: is the app's ball grade right? (31 Oct, worker, local $0, nothing on Modal)

**What it is.** Every match gets `summary.ball_grade` (ipanema/run.py). The app uses it:
`possession_ok = false` hides possession, losses, pass completion, field tilt and the pressing screen
(Vallentuna today); `events_ok = false` gives events the "Beta - low confidence" badge (every match today).
Nobody had checked whether the grade itself tells a good ball from a bad one.

The old rule (27 Sep):
- `near_frac` = share of ball frames with a **visible** player within 4 m of the ball
- `accuracy` = checked frames (clicks) where our ball is within 30 px of the click (needs >= 10)
- possession OK = accuracy >= 0.6; with no clicks: near_frac >= 0.85 and a ball in > 40% of frames
- events OK = accuracy >= 0.6 **and** near_frac >= 0.85

**Test.** Metrica's 2 free pro games, 4 halves, through the pipeline's own steps (tools/metricalab, like the
nightly), 6 kinds of ball, 2 seeds: perfect; our clips' kinds of noise (flicker, streak, veo = 65% of players seen);
and two made-up bad balls the grade must catch - **garbage** (a random spot every 2 s) and **sticky** (the dot
parked on a random player of either team for 2 s, so the team on the ball is a coin flip). Accuracy proxy = 40 random
frames, ours within 2 m. Quality = share of frames where our possession names the right team.
`tools/n0_31a_ballgrade.py`, numbers in `metrica_ballgrade.json`.

| ball | near_frac | accuracy (40) | possession right, frames | old: events OK | old: possession OK without clicks | new: possession OK |
|---|---|---|---|---|---|---|
| perfect | 0.79-0.86 | 1.00 | 94-96% | 2/4 | 2/4 | 4/4 |
| flicker | 0.81-0.88 | 0.75-0.88 | 75-79% | **6/8** | 6/8 | 8/8 |
| streak (camera view) | 0.58-0.62 | 0.78-0.85 | 83-89% | 0/8 | 0/8 | 8/8 |
| veo (camera view) | 0.54-0.59 | 0.53-0.75 | 85-91% | 0/8 | 0/8 | 6/8 |
| garbage | 0.11-0.14 | 0.00-0.03 | 37-52% | 0/8 | 0/8 | 0/8 |
| sticky | **1.00** | 0.00-0.10 | 47-53% | 0/8 | **8/8** | 0/8 |

**What it says.**
- "Near a player" is not a quality test. A perfect ball only scores 0.79-0.86 with all 22 players, and ~0.60 with
  the players our camera shows (our matches: SFK-BP 0.47, AIK 0.50, Vallentuna 0.62). The ball that flickers
  between players scores *higher* than a perfect one, and the coin-flip ball scores 1.0.
- So the old events bar (0.85) can never pass a good ball on our footage, and on pro data it passes the flickering
  ball more often than the perfect one. Without clicks, the old possession rule passed the coin-flip ball every time
  and a perfect ball half the time.
- The clicked accuracy works: every bad ball fails (0-0.10), every realistic ball passes, except 2/8 'veo' runs at
  0.53-0.55 (40 clicks at a 2 m bar is a noisy measure; not changed here).

**Change (new default, `IPANEMA_BALL_GRADE=old` = old rule).** `run.ball_grade()`:
- possession OK = clicked accuracy >= 0.6; no clicks = not verified (withheld)
- events OK = possession OK **and** a by-eye event check of that match passed (`overrides/<match>_events_check.json`
  `{"passed": true}`); none exists yet, so events stay "low confidence" everywhere
- `near_player_pct` stays in the summary as information only; the summary gets `rule`.

**Effect on the app's matches:** none today - SFK-BP (0.82) and AIK (0.67) keep possession OK, Vallentuna (no
clicks) stays withheld, events stay low confidence on all three. What changes is that a future match can't get
possession OK from a ball that just sits on players, and events can turn OK once someone has checked them by eye.
Reaches the app at the next run. Which by-eye check should count for events: Waiting for Daniel.

Tests: tests/test_n0_31a_ballgrade.py (6). Full suite on this machine: 819 pass, 3 fail only because
`supervision` isn't installed here (the same 3 fail without this change).
