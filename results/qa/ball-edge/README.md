# BALL-EDGE (23 Oct 12:25-14:45 UTC, worker, local + free runner, $0, nothing on Modal)

**Question (from N0-21e):** the ball finder's tile-edge rule (`IPANEMA_BALL_EDGE=1`: drop guesses cut by a tile's inner side) removed ~3% of guesses, nearly all junk, on single key frames. Does it make the app's ball better over a whole clip, once it goes through the picker?

**Nightly card 23 Oct:** worse than 22 Oct = nothing.

## How (no Kaggle, no Modal)
- The whole-clip re-detection was ~14 h of CPU, too long for one free runner job. Split into **16 free-runner jobs in parallel** (8 slices x 2 clips, `tools/ball_edge_shard.py`, 68-100 min each). Each kept every tile guess before merging (`raw/*.npz`, 4.5 MB).
- `tools/ball_edge_grade.py` (local): both arms merged from the **same** tile guesses (edge off / on) and fused with the **same** WASB guesses, then the app picker. WASB guesses are read back out of the app's fused inputs (only the fused list is saved). For SFK-BP the real WASB files exist, so the grade also ran with those (exact fuse).
- CPU re-detection is not bit-identical to the app's GPU run: 98% of re-detected guesses above 0.3 confidence are in the app list, 68% below 0.08. Both arms share this drift, so the old/new comparison is fair. The absolute numbers differ a little from the app's.
- `tools/push_retry.sh`: 8 tries with a random wait (was 3), because 16 jobs finish close together. All 16 pushed.

## Result: no change on any key

| | app inputs | old | new (edge on) |
|---|---|---|---|
| SFK-BP 34 checked balls (read-back WASB) | 29 | 28 | 28 |
| same, exact WASB t2x2 | | 29 | 29 |
| same, exact WASB lp30 | | 27 | 27 |
| B4 ball moments /285 (read-back / t2x2 / lp30) | 284 | 283 / 283 / 284 | 283 / 283 / 284 |
| who has the ball /99 (read-back / t2x2 / lp30) | 74 | 71 / 71 / 71 | 71 / 71 / 70 |
| graded passes, real kept /20 | 20 | 15 / 17 / 12 | 15 / 17 / 12 |
| graded passes, fake kept /8 | 8 | 3 / 5 / 1 | 4 / 3 / 1 |
| AIK 39 checked balls | 30 | 27 | 27 |

- The rule drops 2.2% (SFK-BP) / 2.8% (AIK) of finder guesses. The picked ball moves in 41 frames (SFK-BP, 7 spells) and 144 frames (AIK, 18 spells): about 1% of frames. None of those frames is a key frame.
- Fake passes move 3->4, 5->3 and 1->1 depending on the WASB input: pass counts swing like this from tiny input changes (S8). It is noise, not a signal.

## By eye (`diff_sfk_00.jpg`, `diff_aik_00.jpg`, `diff_aik_01.jpg`; red = old pick, green = new pick, the middle frame of each spell)
- SFK-BP 7 spells: new better 2-3 (moves onto a visible ball: #1, #5, maybe #6). Old better 2-3 (old was on the ball, new goes to the keeper / empty grass by the goal: #3, #4, maybe #2). 1 unclear.
- AIK 18 spells: new better ~5 (onto a visible ball, or off a fence post / empty grass: #4, #5, #8, #11, #13). Old better 1 (#0, new on empty grass). The rest are unclear (both at feet, small ball).
- Leaning slightly to the new rule on AIK, even on SFK-BP. It only touches ~1% of frames.

**Verdict:** it does not score better on any key (34 / B4 / who-has-the-ball / passes / AIK 39), so it **stays OFF** (rule: no default unless it scores better). The by-eye lean is small and favours the new rule a little, so switching it on is low-risk but unproven. That call goes under "Waiting for Daniel". Nothing changes in the app.

Tests: `tests/test_ball_edge.py` (5: slices cover the clip, shard dry run, WASB read-back reproduces the fuse, edge arm drops the cut guess but keeps its WASB guess, sheet dry run). Suite 577 pass, 7 skipped.
