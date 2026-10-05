# V3 (5 Oct, worker): is the player on the ball missing because the detector cannot see him in the shade? No.

20 by-eye "who has the ball" moments on Vallentuna (17 from reference/<vall>/ball_key_graded.json + b00/b20/b31),
frames + RF-DETR people boxes from the free runner (tools/v3_frames.py, 2 min CPU, $0), graded by tools/v3lab.py and by eye
on the sheets (results/qa/v3/sheets/<id>.jpg: base boxes green, extra boxes from lower thresholds yellow/cyan/magenta,
export dots red = dark SFK, blue = red Vallentuna). Numbers: results/qa/v3/v3lab.json.

- Detector: the player on the ball is boxed in 20/20 at the pipeline setting (conf 0.3), median conf 0.87, lowest 0.57
  (c05). Including c19, the dark player in the deepest shade (0.88). Lower threshold, shade lifting (gamma 0.6) or local
  contrast (CLAHE) find no extra carrier and add 2.6-5.7 extra boxes per frame (mostly spectators/bench). No detector change.
- Kit model (fitted like the app on the 36 Vallentuna frames): with the 5 Oct hue mode the carrier gets the right team
  20/20; the old reading 14/20 (6 reds called 'neither').
- The app export (4 Oct, before hue mode): carrier present in the right team 13/20; wrong team 3 (c03, c37, c38 - the
  hue-mode model has all 3 right); missing 4: b31 (old reading 'neither', hue mode fixes), c19 (the calibration puts his
  feet 68 m across a ~64 m pitch, so he is dropped as off the pitch), c05 and b20 (kit reading right; b20 is a second-half
  frame where only 3 of 13 people boxed reach the export - cause not traced, needs the tracking rows of that piece).
- So for Vallentuna possession the gap is the kit reading (fixed in code by hue mode, reaches the app with the K3
  re-track, EUR 5, waiting for Daniel) plus a few calibration / tracking drops - not detection in shade.
