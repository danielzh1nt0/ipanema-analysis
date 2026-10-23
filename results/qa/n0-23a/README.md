# N0-23a (23 Oct, worker, local + free runner, $0, nothing on Modal)

**Card:** no 23 Oct card yet (00:20 UTC, before 03:00); the 22 Oct card says worse than 21 Oct = nothing.
M2 / PC2 / BALL-EDGE skipped (main session / waits for M2 / needs a Kaggle GPU; this worker runs only local + the free runner).

**What was looked at:** the transition figure in `stats.json` teams[]: `forward_within_3s_pct` ("forward pass within 3 s
of winning the ball") and `time_to_forward_pass_median_s`, plus the per-event `forward_within_3s` on each ball-won event.
Nobody had checked it.

## How it was checked (no clicking)
- Truth = Metrica's free pro games (2 games x 2 halves): after each labelled BALL LOST, the winner's first completed PASS
  that gains >= 5 m towards the goal, timed from the winner's RECOVERY to the pass's end (`tools/n0_23a_transition.py`).
- Ours = the app's own steps on Metrica's positions (`tools/metricalab.ours`), perfect ball and our 'streak' and 'flicker'
  ball noise (2 seeds each). Per-event agreement on the wins we and the truth both have (same team, within 2 s).

## Findings
1. **Harness bug (fixed):** `tools/metricalab.ours` used a fixed attack direction (A ->). That is backwards in 2 of the 4
   Metrica halves (Game 1 2nd, Game 2 1st), so every forward pass there read as backward. Now keepers, else the
   player-position fallback, like `ipanema/run.py`. Passes/possession counts in the nightly card don't use the direction.
2. **Clock started too late (fixed, on):** the counted state turns to the winner only once he holds the ball - often the
   receiver of his team's FIRST pass, so that pass was never seen (e.g. Game 1, 2nd half, 2573.6 s: #7 intercepts, passes
   8 m forward to #14; the state turns at #14). Now the search and the clock start at the winner's first touch after the
   loser's last control (`IPANEMA_FWD_FROM_WIN=0` = old). The win time used by every other figure is unchanged.

| Metrica, 4 halves | per-event agree | true 'yes' found | false 'yes' | median time vs truth | half % off truth |
|---|---|---|---|---|---|
| perfect ball, old | 116/153 | 19/41 | 15 | +1.24 s | 4.1 pts |
| perfect ball, new | **129/153** | **25/41** | **8** | **+0.42 s** | 6.4 pts |
| flicker noise, old | 173/238 | 29/51 | 43 | +0.61 s | 8.5 pts |
| flicker noise, new | 177/238 | 28/51 | 38 | +0.54 s | 9.6 pts |
| streak noise, old / new | 36 / 36 of 49 | 3 / 1 of 10 | 6 / 4 | +0.92 / +0.79 s | 9.2 / 4.8 pts |

Picture: `forward3_vs_truth.png`. Rows: `metrica_transition_old.json`, `metrica_transition_new.json`.
- With a perfect ball the new clock is clearly better per event and on the time. The half-level % is not closer: our
  wins are only the kept ones (S9 rule), a different population from all truth wins, so the % matching before was partly luck.
- **Under our noise neither version is good:** about as many false 'yes' as real ones found; the half % reads 19-36% where
  the truth is 14-25%. The figure is limited by the ball, like 'regained within 5 s' (N0-22f).

## Our app clips (exact picker inputs, `tools/n0_23a_clips.py`, `clips.json`)
- SFK-BP clip: 17 wins, forward within 3 s 29% -> 24%, median 1.35 -> 2.69 s. AIK clip: 17 wins, 41% -> 35%, 2.05 -> 3.92 s.
  The answer changed on 4 of 34 wins.
- Picture sheets (free runner, `sheet_*.jpg`, 7 moments: the 4 changed + 3 with a 'pass' under 0.5 s). Not blind (the
  old/new numbers were already known). Looked at sfk00, sfk02, aik05: around the win the yellow ring is mostly NOT on a
  visible ball - in the air (+1.5-3.5 s on sfk02), on a fence post for 3 s (aik05 +2.4-5.2 s), on empty grass by the far
  touchline (sfk00 +1.5-4.0 s). So on our footage neither the old nor the new answer can be confirmed; the quick
  0.2-s 'passes' come from the ball dot, not from play.

## Change
- `ipanema/possession.py` `FWD_FROM_WIN` (default on; `IPANEMA_FWD_FROM_WIN=0` = old). Reaches the app at the next run.
- `tools/metricalab.py`: attack direction like run.py; returns `attack_right`.
- `tests/test_fwd_from_win.py` (4: first pass counted from the first touch, old misses it, backward pass not counted,
  metricalab direction).
- Question for Daniel (with the N0-22f one): the 'forward within 3 s' % is ball-limited too - hide it / mark it Beta?
