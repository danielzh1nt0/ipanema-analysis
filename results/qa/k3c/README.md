# K3c (5 Oct): Vallentuna team colours, third try

- K3 (hue mode + per-player classifier): sunlit SFK (black) players became Vallentuna (QA t=638 s: A 2 / B 13).
- K3b (hue mode, colour only): now nearly everyone became SFK (t=1045 s: A 17 / B 1; possession 79/21; owner check 9/17).
  The median a* of a sun-washed or blurred red shirt is 8-19, black shirts with pink numbers reach 17, the rule's middle was 19.5.
  Piece videos are NOT the cause: full vs piece frames differ by < 0.3 Lab units and give the same labels (k3c_piece_frames.py, free runner).
- K3c: in hue mode, A/B from the share of strongly kit-coloured shirt pixels (>= 25% red kit; 10-25% red unless >= 8% dark pixels).
  Ball carriers (20 by eye): 18/20 right (median rule 12/20). All 389 people in the 22 frames: label_A_dark.jpg / label_B_red.jpg,
  about 10 of 181 and 12 of 149 in the wrong sheet by eye (per detection; tracks vote over 25 checks).
- Only Vallentuna uses hue mode (SFK-BP, AIK: black vs white/grey).

## Free route (Daniel: no more paid runs), 5 Oct evening
- Kaggle free GPU (kaggle/vall_relabel.py, 64 min): every 2nd exported frame, people detected, each exported player's box
  labelled with the K3c rule, majority per player id -> overrides/p15u-vs-vallentuna-2026-10-03-6cce_teams.json.
  1068 of 5573 ids change; players per frame A 6.9 / B 7.0 (app now ~10 / 4); owner check 13/17 (app now 9/17).
- By eye on the 12 QA moments (relabel_check/*.jpg, red dot = SFK, blue = Vallentuna): near players almost all right, far
  players and players whose track switched between people often wrong; roughly 3 of 4 right overall (app now about half).
  t=638 is the worst (~7/14), t=1045 / 4628 / 5035 good (12-14 of 14-17).
- Not in the app: needs one join on Modal (CPU, ~EUR 0.10) which reads the override file. Waiting for Daniel's go.
