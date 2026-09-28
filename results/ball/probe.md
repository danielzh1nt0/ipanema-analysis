# B2: do the finders see the missed balls? (28 Sep, GPU 7.5 min, Daniel: go)
Every finder variant at a very low cut-off; "seen" = a guess within 30 px of the true ball.

| | 34 clip moments | 81 exam frames with a ball |
|---|---|---|
| click finder as in the app (0.05) | 18 | 54 |
| click finder, cut-off 0.005 | 26 | 64 |
| click finder, 2x2 tiles | 26 | 64 |
| WASB, all peaks >= 0.05 (not capped at 6) | **31** | **79** |
| WASB, cut-off 0.01 | 33 | 80 |
| any variant | 33 | 81 |
| WASB top guess = the ball | 26 | 56 |
| WASB: ball within its top 10 guesses | 27 | 76 |

Findings:
- The finders DO see the ball almost always (WASB 31/34 and 79/81 at the normal cut-off). The earlier "7 of 34 never seen"
  was caused by our code keeping only WASB's 6 strongest peaks per frame (app cache: exactly 6 per frame).
- When WASB's top guess is not the ball, the ball is usually far down its list (clip: rank 16-35 of ~44 guesses).
  So the problem is SCORING/CHOOSING among WASB's guesses, not seeing. -> a second-opinion scorer (T1) on WASB's guesses
  is the most promising next step; the exam says the ball is in WASB's top 10 in 76/81 frames.
- Going below 0.05 adds little (+2 clip, +1 exam) and doubles the guesses.

Also (B8, tools/fusedlab.py, both finders' real guesses copied from Modal): the app's 22/34 is exactly
"round-10 click finder + WASB + old players". The round-10 click finder beat round 1 on the exam (70 vs 66/108) but
is WORSE on the clip (ball among its guesses 17 vs 19/34, 1.4 vs 2.4 guesses per frame). With new players: 23/34.
"Ball with this player" option on both finders: 23-26/34, within noise of 34 moments; not proven.
