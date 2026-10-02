# S4: are our passes real? (2 Oct, SFK-BP clip, free)

The clip's 87 exported passes were reproduced exactly offline (tools/passlab.py) and the first 30 checked by eye on
3-frame strips (results/kaggle/passcheck/pass_00-04.jpg): **20 real, 8 fake, 2 can't tell** -> about 30% of our passes
are not passes, which fits the full-match counts (16-20 per minute, 1.5x a plausible rate).

What the fakes are: the ball lying still between two standing players (a stoppage or a player holding the ball) while
the ball picker and the carrier flicker between them (#1-4: four "passes" in 2 s at 10.8-12.6 s), a ball pick off the
pitch (#0 on the fence, #20 in the trees), and duplicates of one contact 0.2 s apart (#21/22, #25/26).

Rules tried on the exact inputs (none kept): ball must move >= 5 / 8 m between contacts (the ball picks jump so much in
metres that fakes pass and reals fail: real kept 17/20, fake kept 8/8); one pass per second (fake 5/8 but real 16/20);
with the possession state (fake 2/8, real 13/20). Pixel features (ball staying at the passer's feet, camera-corrected
ball travel) do not separate them either: in the fake windows the PICK moves although the ball does not.

Conclusion: the fake passes come from the ball picker, not the pass rule - a still ball must stay locked in place and a
pick must not leave the pitch. That is picker work (B4c-style still-ball logic + an off-pitch veto), not for the demo.
Pass COUNT stays hidden in the app; pass completion % and the pass map are kept (they are made from the same passes, so
they carry the same ~30% noise - say so if asked).
