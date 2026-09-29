# P7 (29 Sep): people off the pitch on grounds without calibration

Checked on the 24 saved frames per ground (results/qa/kitprobe/), by eye. Tool: `tools/p7lab.py` (free, local).

## What was wrong
- The P4 "feet on grass" test (`kits.on_grass`) compared the grass under a person's feet with the near-side grass.
  At night the floodlit middle of the pitch is much brighter, so it failed for most real players:
  at Reymersholm it dropped 321 of 414 people, and kept people standing on the dimmer grass behind the far line.

## New test: the pitch's top edge (`kits.pitch_top`, `kits.feet_on_pitch`)
- Finds where the grass that reaches the bottom of the picture ends (colour mostly, lightness ignored), per column.
  Feet must be at least ~1% of the picture height below that edge. No calibration needed.
- Reymersholm (night): keeps 305 of 414, drops 109. By eye: about 7 real players dropped (standing right on the far
  line, or a keeper in front of his goal) and 3 non-players kept (spectators right at the bottom). Pictures: `*_edge_*.jpg`
  (green = counted, red = dropped, magenta = pitch edge). Also looks right on Spånga and on SFK in daylight.
- Used only on grounds with no calibration (`tracktest.py`: `IPANEMA_OFFPITCH`, on by default when not SFK):
  a person read off-pitch is not counted that frame, and a track that is off-pitch half the time is dropped.
- Dry run (tracktest with a stand-in detector = the saved boxes): dark per frame 11 -> 8, light 4 -> 3.
  So the off-pitch people were most of the "dark 10 / light 3" gap; the rest is the colour problem below.

## Whites read as green at night: NOT fixed
- The old check (62/71 right, 7/31 whites read green) was on the 93 people the broken test kept, mostly on the far edge.
  Claude labelled 222 more on-pitch people by eye (`results/qa/kitprobe/reym_labels_p7.json`): on all 252 players
  the kit reading is right for **195/252 (77%)**, and **51/115 whites read green** (blurred, small, floodlit players).
- Tried, none good enough to change the default (numbers in `p7lab.json` and below):
  | kit learning | right / 252 | whites read green | greens read white |
  |---|---|---|---|
  | current (learn from 'on grass' people, two biggest colour groups) | 195 | 51 | 4 |
  | learn from all on-pitch people (edge test) | 149 | 7 | 48 (greens split into dark/lit halves) |
  | edge + pick the most different big group as the 2nd team (`IPANEMA_KIT_PAIR=far`) | 200 | 47 | 5 |
  | local grass colour around each player / no grass removal / grass share | 122-202 | 44-52 | 4-67 |
- The misread whites have a torso colour between the two kits (blurred shirt + grass). A colour-only fix won't do it;
  next step is a small per-player classifier trained on these 252 labels + other grounds, or using more of the body.

Defaults unchanged for kit learning (`IPANEMA_PITCH_TEST=grass`, `IPANEMA_KIT_PAIR=big`); the new ones are options.
