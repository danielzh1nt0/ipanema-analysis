# H1: outside ball pictures (martinjolif/football-ball-detection) - 29 Sep

**Short answer:** clean labels and the ball is the same size as ours, but it is all Bundesliga TV footage, and the source footage probably isn't free to use. Worth one free test only if the licence is OK.

## What is in it
- 1,237 pictures (train 989 / valid 123 / test 125), all 1920x1080, exactly one ball each. YOLO format, one class "ball".
- The ball is about as small as ours: median 12 px wide (10% under 9 px, 10% over 18 px). Our Veo training box is 14-18 px. 99% of their balls are Veo-sized or smaller.
- Checked 48 random pictures by eye (sheets `results/free/h1/sheet_00-03.jpg`, zoomed ball on the right of each): **48/48 boxes on the ball**, no wrong labels seen.

## How it differs from our footage
- All pro TV main camera: big stadiums, perfect grass, even light, camera following play from the middle. No fences, no amateur grounds, no night games, no Veo panorama stretching at the sides.
- The ball is almost always on the ground in open grass or at a player's feet. Few hard cases (ball in the air, in a crowd, far side) - exactly where our finder misses (B5: 8/13 misses at feet/crowd, 4/13 in the air).
- Frames come from short clips (file names `08fd33_0_mp4`, `0a2d9b_0_mp4`, ... ~8-13 frames each), so many pictures are near-copies. Their train/test split is by frame, not by clip.
- Compare with our own balls at the same zoom: `results/free/h1/veo_reference.jpg`.

## Licence (needs Daniel)
The Hugging Face / Roboflow page says CC BY 4.0, but the clip names match the videos of the Kaggle "DFL Bundesliga Data Shootout" competition. That competition's rules limit how its videos can be used. Whoever uploaded the frames can't change the licence of the footage. So using it in a product we sell is doubtful; for testing only it is lower risk.

## Would it help?
Probably a little at most: our new ball finder (RF-DETR) was already trained on ~5,000 ball pictures from our own grounds, and these add easy TV balls, not the hard cases. The honest test is one free Kaggle training run (current training set + these 1,237, same settings) graded on the exam (now 84/108) and the 34 clip moments. Not started: this worker only uses the free GitHub runner, and the licence question comes first.

Files: `results/free/h1/` (stats.json, sheets, files.txt, README.md = the dataset card). Code: `ipanema/hfball.py`, `tools/h1_hfball.py`, test `tests/test_hfball.py`.
