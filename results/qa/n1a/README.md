# N1a - floodlit near whites removed as referee / staff on Solberga (6 Oct, worker, local + free runner, $0, no Modal)

**Cause.** Under the floodlights the white players nearest the camera read much lighter than the white team's centre
(torso L ~97 vs ~73, colour almost the same), so the kit reading called them "neither team" and tracking removed them
with the referee / staff. On 4 Oct the screen-edge keeper rule rescued #5 by luck; P2e switched that rule off, so on
5 Oct light players went 7 -> 6 per frame.

**Fix.** `kits.bright_team`: a "neither" reading whose colour (a*b*) is near the LIGHTER kit's colour (within 0.6 x
the colour gap between the two kits, nearer that kit than the other) and is at least as light as that kit joins it.
Only on grounds without calibration (`KitTeamModel.offpitch`, tracktest); the app's run sets offpitch off, so the app
is untouched. Env `IPANEMA_KIT_BRIGHT` (0 = old). A darker referee or a referee in another colour stays out.
A first version that also let readings join the DARK kit moved red players (Vasalund), striped players (Spånga) and a
goal post into the black team - caught on the sheet and removed.

**Offline, all 18 P2 pieces** (`tools/n1alab.py` -> `n1alab.json`, sheets `changed_0.5.jpg`, `changed_0.6.jpg`,
`team_reference.jpg`):

| setting | people that change | by eye |
|---|---|---|
| 0.5 | 12 (Solberga 1500: 9, Spånga 1159: 3) | 11 right, 1 can't tell |
| **0.6 (default)** | 20 (Solberga 1500: 12, Spånga 1159: 8) | **19 right** (12 near whites, 7 yellow-striped players), **1 can't tell** (white shirt, blue socks on the halfway line - maybe an official) |

Every other piece unchanged; Reymersholm 252 labelled night players (204 / 188 / 220 by piece model) and the 88
labelled people on 4227 (47/64, referee/keepers out 12/12) unchanged.

**Free-runner re-track** (`tools/n1a_track.py`, same 20 s as P2; dry-run first with a stand-in detector) and the 6 Oct
nightly (which already ran with this code):

| Solberga 1500, light players per frame | |
|---|---|
| 4 Oct (keeper rule rescuing #5) | 7 |
| 5 Oct nightly (N1) | 6 |
| 6 Oct nightly / N1a re-track | **8** / **8** |

By eye (`solberga_before_after_0000.jpg`, `_0256.jpg`; top 5 Oct, bottom now): #5 on the right AND the white on the
left now have their blue ring; the dark referee in the middle has none. Spånga 1159: the same numbers as P2 (dark 9 /
light 7; the track vote already kept those players). Nightly card 6 Oct: "Worse than 5 Oct: nothing".

**Still wrong (not new):** at frame 256 a person in a blue shirt near the centre (the referee or a staff member) has a
blue (white team) ring - the same on 5 Oct, before this change.

Tests: `tests/test_n1a_bright_kit.py` (6). Suite here: 291 passed, 1 failure that also fails without this change
(test_ballscorer needs torch, not installed on this machine).
