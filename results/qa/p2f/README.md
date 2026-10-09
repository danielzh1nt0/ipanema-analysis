# P2f - Reymersholm 4227: dim far whites read as the green team (9 Oct, worker, local + free runner, $0, no Modal)

**Cause.** Under the floodlights the far white shirts read cream (torso L 42-65, a* -4..-12, b* +10..+26). In lightness
and colour together that is nearer the green team's centre (dim, L 46) than the bright white one (L 86), so 13 of 40
labelled whites read green on the key frames and 15 of 40 after tracking (P2e). Lightness can't separate them (greens
L 22-62 too). The **hue** does: the cream readings lean from the green kit's hue towards yellow (98-124 deg; greens
122-172; the green kit's core hue 137).

**Fix** (`ipanema/kits.py`: `core_hue`, `dim_light_team`, used in `KitTeamModel._colour_lab`). When one kit is light and
colourless and the other is a COOL colour (green / blue / purple), a reading in the coloured kit whose hue leans at
least 18 deg towards yellow joins the light kit. Uncalibrated grounds only (`offpitch`, tracktest); the app is untouched.
Env `IPANEMA_DIM_LIGHT` (degrees, `0` = old). Test: `tests/test_p2f_dim_light.py`; suite 327 passed.
A first version without the cool-kit gate moved 8 orange Solberga players and 5 red Djursholm players into the white
team (a warm kit itself turns towards yellow under warm light) - caught on the sheet, fixed.

**Offline** (`tools/p2flab.py` -> `p2flab.json`; kit model on each P2 piece's 8 key frames):

| | before | now |
|---|---|---|
| 4227, 88 labelled: team right | 47/64 | **59/64** |
| 4227: whites read green | 13/40 | **1/40** |
| 4227: greens read white | 1/24 | 1/24 |
| 252 labelled night players (4227 model / 2227 model) | 204 / 188 | **217 / 199** |
| Reymersholm + Spånga kitprobe keys, SFK-BP key frames | 236/252, 242/279 | identical |
| other 15 P2 pieces | - | 0 people change |

People that change (`changed_people.jpg`, `changed_2227_context.jpg`): 4227 12, all labelled whites now right;
2227 4: 2 whites now right, 2 greens now wrong (one lying on the ground, one with a yellow captain's armband).
Threshold sweep 10-30 deg on 4227: 16-18 best (59/64); 18 kept for the margin to the greens.

**Free-runner re-track of 4227** (`tools/p2f_track.py`, same 20 s as P2-P2e, graded with `tools/p2e_grade.py`):

| key frames, 88 labelled | P2e | now |
|---|---|---|
| whites read as green | 15/40 | **3/40** |
| whites right | 25/40 | **37/40** |
| greens right | 23/24 | 23/24 |
| referee / keepers read as a team | 2/12 | 2/12 |
| spectators read as a team | 3/11 | 3/11 |
| per frame dark / light (median) | 5 / 3 | 4 / 5 |
| light: median time tracked | 4.25 s | 6.07 s |

By eye (`track/.../compare_*.jpg`): frame 427 all whites blue, all greens red; frame 256 the near white is now white,
two far whites by the goal area still green (tracks 2, 3: their readings stay green-leaning in that stretch).
**Left:** those 3 far whites, the pink referee at the start of his track (frame 256) and the fence spectators.
