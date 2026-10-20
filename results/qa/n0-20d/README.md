# N0-20d (20 Oct 16:20 UTC, worker, local, $0, nothing on Modal)

**Nightly card 20 Oct:** worse than 19 Oct = nothing. No new main-session code since 12:20, so I looked for untested app code (coverage run: suite 488 pass; `tracking.py` 73%, the per-track team vote inside `track()` had no direct test).

## Finding: a tied team vote was decided by chance
Every team vote used `max(set(v), key=v.count)`: the per-track team in tracking (last 25 colour readings), a goalmouth keeper's kit, a track's team in the stats (`analytics.stats`), and the T1 relabel smoothing (`fullmatch.team_segments`). On a tie (e.g. 2 A / 2 B readings) this returns whichever label the set lists first, and Python shuffles that order **per run** (salted string hashes). Same video, same code: A in one run, B in the next (shown in `tests/test_vote_ties.py`: 12 runs give more than one answer).

How often (replayed on the saved rows of 35 free-runner tracking checks, 270k player-frames, `tools/n0_20d_ties.py` -> `ties.json`): **0.5% of player-frames were a coin flip**, up to 2.9% on a piece (Spånga 1159, Reymersholm 4227 after P2f); 165 of 1,339 tracks hit a tie at least once. Approximate: frames where a track was skipped (off pitch / referee) are not in the saved rows.

## Fix (on by default, `IPANEMA_VOTE_TIE=hash` = old)
New `ipanema/vote.py`: `majority()` breaks a tie by the label read **last** (or by `prefer` if given and tied); `team_vote()` is the tracking vote moved out of `track()` unchanged apart from the tie. Without a tie every answer is the same as before (5,000 random votes in the test). T1 smoothing: a tied 4-read window keeps the read itself.

Two tie-breaks tried, graded blind by eye on every tie moment that falls on a saved key frame (14 moments, `tie_moments_blind.jpg`, graded before looking at the answers; 4 can't tell - keepers / overlapping players):

| tie goes to | right | wrong |
|---|---|---|
| latest reading (**used**) | 7 | 3 |
| track's team so far (tried) | 4 | 6 |
| old run (chance) | 6 | 4 |

The latest reading wins because a tie usually means the track has slid onto another player (ID slide) and the fresh reading sees the new one. Cost: team changes along tracks 424/400 (old, either set order) -> 471 on the replay (+12%); keeping the team so far would give 357 but is wrong more often. Small sample, so the main gain is the same answer on every run, not accuracy.

Reaches the app at the next tracking run (tracking vote) and the next join (stats team, keeper kit). Tests: `tests/test_vote_ties.py` (7), `tests/test_track_vote.py` now calls the moved vote, `tests/test_t1_segments.py` compares with the old inline rule under `IPANEMA_VOTE_TIE=hash`.
