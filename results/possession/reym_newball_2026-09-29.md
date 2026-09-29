# Reymersholm who-has-the-ball with the new ball finder (29 Sep, free)

New ball guesses on the 5-min piece (kaggle/ball_reym.py, every 2nd frame, 26 min on Kaggle), fused with WASB, same picker.
Score on the 32-moment key (results/review/who_answers_reym.json), near 1.5 m: **23/32 new vs 27/32 old**.

The 4 moments that flipped (5230, 6845, 7579, 8754) were pulled as full pictures (results/kaggle/reym_frames/).
In at least 2 of them (6845, 8754) the real (orange) ball is visible near the NEW pick 1 s before/after, not at the old
ball's spot. The key was graded by eye (Claude) from strips that showed the OLD ball's circle, so it leans towards the
old ball. Verdict: this key can't judge the new ball. Not a proven regression, not a proven gain.

Fix: re-grade the Reymersholm key blind (strips with no ball marker), then score both balls again.
