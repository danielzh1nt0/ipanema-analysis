# Stats: balls lost / possession on Metrica pro data (28 Sep evening, free)
Answer key: Metrica's 2 free pro games, hand-labelled BALL LOST events (490 in 4 halves). A loss counts as found if ours is
within 2 s and the same team. Tools: tools/turnoverlab.py, tools/speedlab.py.

1. **Turnover rule too strict.** A loss only counted if the team had the ball >= 3 s before AND the winner kept it >= 3 s.
   Perfect positions: found 100/490 (20%), 172 reported. With 1 s / 1 s: 181/490 (37%), precision ~55%.
   Now adjustable (min_before_s, min_after_s); default unchanged.
2. **Possession model broken by position wobble.** Ball speed was measured frame to frame, so 0.7 m of wobble reads as
   ~20 m/s = 'in flight' = nobody has it. With realistic noise: 'nobody has the ball' 76-87% of the time, losses found 0-5
   per half. Measuring speed over 0.6 s: 'nobody' 46-57%, losses found 16-37 per half, possession error 7 -> 5.5 pts avg.
3. **But on our real SFK-BP clip** the windowed speed raises 'nobody has the ball' from 25% to 35-62% (real picks have
   single-frame jumps). Median-smoothed version (speed_win_s negative) is better but still above 25%. We have NO answer key
   for who has the ball on our own footage, so we cannot tell which is right. -> option added, default unchanged.
4. Next: a who-has-the-ball answer key on our own footage (E1): short clips with the ball marked, graded by Claude by eye,
   not single stills (tried: a still at this distance is not readable).
