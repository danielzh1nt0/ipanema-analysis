# K3c (5 Oct): Vallentuna team colours, third try

- K3 (hue mode + per-player classifier): sunlit SFK (black) players became Vallentuna (QA t=638 s: A 2 / B 13).
- K3b (hue mode, colour only): now nearly everyone became SFK (t=1045 s: A 17 / B 1; possession 79/21; owner check 9/17).
  The median a* of a sun-washed or blurred red shirt is 8-19, black shirts with pink numbers reach 17, the rule's middle was 19.5.
  Piece videos are NOT the cause: full vs piece frames differ by < 0.3 Lab units and give the same labels (k3c_piece_frames.py, free runner).
- K3c: in hue mode, A/B from the share of strongly kit-coloured shirt pixels (>= 25% red kit; 10-25% red unless >= 8% dark pixels).
  Ball carriers (20 by eye): 18/20 right (median rule 12/20). All 389 people in the 22 frames: label_A_dark.jpg / label_B_red.jpg,
  about 10 of 181 and 12 of 149 in the wrong sheet by eye (per detection; tracks vote over 25 checks).
- Only Vallentuna uses hue mode (SFK-BP, AIK: black vs white/grey).
