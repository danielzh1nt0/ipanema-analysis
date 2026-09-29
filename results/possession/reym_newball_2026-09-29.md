# Reymersholm who-has-the-ball with the new ball finder (29 Sep, free)

New ball guesses on the 5-min piece (kaggle/ball_reym.py, every 2nd frame, 26 min on Kaggle), fused with WASB, same picker.
Score on the 32-moment key (results/review/who_answers_reym.json), near 1.5 m: **23/32 new vs 27/32 old**.

The 4 moments that flipped (5230, 6845, 7579, 8754) were pulled as full pictures (results/kaggle/reym_frames/).
In at least 2 of them (6845, 8754) the real (orange) ball is visible near the NEW pick 1 s before/after, not at the old
ball's spot. The key was graded by eye (Claude) from strips that showed the OLD ball's circle, so it leans towards the
old ball. Verdict: this key can't judge the new ball. Not a proven regression, not a proven gain.

Fix: re-grade the Reymersholm key blind (strips with no ball marker), then score both balls again.

## E6 blind check (same evening)
Old and new ball agree (within 30 px) on 40 of the 60 moments. On the 20 where they disagree, zoomed crops of both picks
were shown side by side in random order (A/B), graded by eye BEFORE opening the key (results/review/ball_ab_reym_2026-09-29.json,
pictures results/qa/ball_ab_reym/):

| Ball actually at | Moments |
|---|---|
| **new pick** | **10** |
| old pick | 3 |
| neither | 1 |
| can't tell | 6 |

Verdict: on Reymersholm (night, never trained on) the new ball is right far more often where the two differ. The 23/32 vs 27/32
possession score came from the biased key. The possession key itself should be re-graded from blind pictures before it is used again.
Old pick errors seen: a floodlight is not in this set; the NEW finder's wrong picks included a floodlight (m22) and leaves (m26).
