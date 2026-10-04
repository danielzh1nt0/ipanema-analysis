# P2d - Reymersholm 4227: spectators by the right-hand fence read as the green team (4 Oct, worker, local + free runner, $0, no Modal)

**Cause.** The spectators stand on the green run-off strip beyond the white boundary line by the right-hand fence, so
both pitch tests (grass at the feet, pitch top edge) keep them as players.

**Fix** (ipanema/kits.py `side_lines` / `feet_inside_sides`, used only where the off-pitch test already runs: grounds
with no calibration, i.e. tracktest; the app keeps it off, `IPANEMA_OFFPITCH=0`). A boundary side line is a long,
bright, colourless, steep straight line with grass on the pitch side, less than 20% of the pitch beyond it, and NOT
mostly grass right beyond it (band ~25-160 px past the line < 65% grass: path, fence, net). People whose feet are
beyond it read off the pitch. The last check is what tells a boundary from the halfway line seen at the picture's
edge: beyond the halfway line it is 75-100% grass, beyond the 4227 boundary 21-61%. A first version without it dropped
real players beyond the halfway line on Djursholm, SFK-BP and Reymersholm 2227 - caught by eye on the sheets.
Env `IPANEMA_SIDE_LINES` (0 = old). Test: tests/test_p2d_side_lines.py.

**Offline check** (tools/p2dlab.py -> p2dlab.json; same kit model, side test off vs on, on the key frames of all 18 P2
pieces + the Reymersholm / Spånga / SFK-BP kitprobe frames):

| | before | now |
|---|---|---|
| 4227, 11 labelled spectators/staff kept out | 0/11 | **10/11** |
| 4227, players dropped as off the pitch | 3/64 | 3/64 |
| 4227, team right (64 players) | 47/64 | 47/64 |
| 252 labelled night players (models of 726 / 2227 / 4227) | 220 / 188 / 204 | 220 / 188 / 204 |
| ...labelled spectators kept out (726 / 4227 models) | 26 / 26 of 39 | 28 / 28 of 39 |
| people that change on the other 17 pieces + 3 kitprobe grounds | - | 10 |

All 20 people that change (`changed_people.jpg`, frames in `frame_*.jpg`, red = side line found, magenta = newly off)
are off-pitch people by eye: the 4227 fence spectators (10), Reymersholm kitprobe fence spectators (2), coaches /
staff standing beyond the near line at Solheim (5), two bibbed people by the Solberga bench (3). No player lost.
Demo matches: untouched by construction (the app runs with the off-pitch test off; covered by the test).

**Not fixed (-> P2e):** the dim far whites (13/40 read green; P2c tried many colour readings, none better) and the pink
referee read green in tracking (on the key frames all 12 keepers/referee read 'neither', so it is the track vote).

**Free-runner re-track** (tools/p2d_track.py, same 20 s as P2/P2b/P2c, track/<piece>/, 16 min, $0): by eye
(compare_*.jpg vs results/qa/p2c/track/..._4227) the fence spectators tracked on the key frames go 11 -> 3: frame 427
3 -> 0, 512 4 -> 0 (dark 9 -> 5, every player kept), 598 4 -> 3 (still in there). Median per frame unchanged (dark 6 /
light 3); the dark team's median track time 4.9 -> 1.5 s, because the standing spectators made the longest 'dark' tracks.
Still wrong as in P2c: the pink referee and the dim far whites read as the green team.
