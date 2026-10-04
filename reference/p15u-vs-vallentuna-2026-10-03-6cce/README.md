Vallentuna answer keys (4 Oct, by eye):
- ball_key_graded.json: 78 moments (38 + 40), the finder's guess that is the ball, and who has it (owner) for 23 of them.
- ball_key_px.json: the same balls as {frame: [x, y]}. Deliberately NOT named ball_gt.json: the pipeline would then grade the
  ball at 0.75 and SHOW possession, but possession is right in only ~52% of the clear moments (the player on the ball is
  often missing: sun-washed players dropped as 'neither team'). Rename to ball_gt.json once the players re-track is done
  and possession checks out.
