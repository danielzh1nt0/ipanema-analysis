# C3: Vallentuna camera coverage 43% (4 Oct, free)

The rows file has a pose for 5,961 of 5,966 seconds, but 2,419 rows are marked `refine doubtful` (the refined fit had a rival
view or a high cost: long shadows, the pitch's extra blue lines) and were thrown away; frames near them got a stale pose.
1,260 of those rows agree forward/backward within 15 px.

By eye (results/kaggle/vall_calib_look, Kaggle): 16 random doubtful-but-agreeing rows drawn on the video -> 16/16 on the real
pitch lines, as good as 8 confident rows. So `refine doubtful` is no longer a veto in linecal.brave() when forward/backward
agree (the other vetoes stay).

Usable rows: Vallentuna 2,962 -> 4,187 of 5,966 (50% -> 70%); SFK-BP 5,244 -> 5,367 of 6,304 (+2%). Takes effect at the next
join of each match (CPU, cents); the demo halves in the app were not re-exported for this.

Real run (4 Oct, pieces re-prepared as v5, CPU 7 min): calibration coverage 0.43 -> 0.70, frozen frames 102,157 -> 53,574;
players/possession unchanged (A 4 / B 5 per frame, 34/66). Still FAIL against the 0.95 bar; the remaining 30% are rows that
disagree forward/backward or have no anchor (fast pans, zoom) - next step would be a Vallentuna-specific line model check.
