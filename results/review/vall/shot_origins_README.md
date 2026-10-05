# V1: shot origins for Vallentuna (5 Oct, 01:30-03:30) - not achieved, honestly

1. By eye on 1-s frames (960 px, results/kaggle/shot_origins): the ball is 2-4 px; the strike moment can't be seen for most
   shots. 2 placed with some confidence (3604 at ~(100.7, 27), 3742 at ~(14.5, 44)), the rest would be guesses.
2. Finder track at 10 fps in each window (results/kaggle/shot_ball_track) + 'last fast move towards goal'
   (tools/shot_strike.py): the top guess jumps 400+ px between frames (spare balls, heads, line marks), so the "strike"
   picked is often after the goal or a jump (speeds of 36-63 m/s). Checked on the candidate sheets
   (results/kaggle/strike_crops): not usable.
Nothing was written to the app. Shots keep x/y = null.
Options: (a) Veo's own shot map (Veo Analytics shows shot spots) -> Daniel screenshots, Claude transcribes; (b) a click page
playing the 8 s around each shot from R2, one click per shot (~10 min per match, anyone can do it); (c) timeline only.
