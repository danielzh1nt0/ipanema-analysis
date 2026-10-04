# P2c - Reymersholm piece 4227: greens read as whites, spectators tracked (4 Oct, worker, local + free runner, $0, no Modal)

**Key.** The 88 people the detector finds on the piece's 8 key frames, labelled by eye (`labels_4227.json`):
24 green players, 40 white players, 12 keepers/referee, 11 spectators/staff, 1 can't tell.

**Cause of the team mix-up.** The per-player classifier (P8, body colour histograms) was switched on for this piece and
it had learned the wrong thing (near/lit vs far/dim, not green vs white): it moved 10 of 24 greens into the white team
and 25 of 40 whites into the green team. Colour alone gets 47/64 right; with the classifier 26/64.
The same happens on Spånga piece 2576: by eye ~26 of the classifier's 28 moves put black players in the striped team.

**Fix** (ipanema/kits.py, `_fit_cls`): a self-check. The classifier is dropped (colour only) when it moves more than 25%
of one team's *clear* colour readings to the other team (clear = much nearer one team colour than the other). Where it
helps it moves 7-18%; on 4227 31-32%, on Spånga 2576 30%. Env `IPANEMA_CLS_MAX_FLIP` (0 = old). Test:
tests/test_p2c_cls_selfcheck.py.

**Offline check** (tools/p2clab.py -> p2clab.json; kit model fitted on each P2 piece's 8 key frames):

| | before (P2b) | now |
|---|---|---|
| 4227, 88 labelled: team right | 26/64 | **47/64** |
| 4227: greens in the white team | 10/24 | 1/24 |
| 4227: whites in the green team | 25/40 | 13/40 |
| 4227 model on the 252 labelled night players | 150/252 | **204/252** |
| Reymersholm 726 / 2227 models on the 252 | 220 / 188 | 220 / 188 (classifier kept on 726) |
| Reymersholm key frames (P8 key, 252) | 238 | 238 |
| Spånga key frames (279) | 242 | 242 |
| Spånga 2576 per frame dark / striped | 6 / 8.5 | 10.5 / 5 (classifier dropped; its moves were wrong by eye) |
| the other 15 pieces, SFK-BP key frames | - | identical |
| demo matches (SFK-BP, AIK, Vallentuna fit/test frames, tools/p2c_demo_check.py) | - | 0 people change |

Sheets: `4227_before_team*.jpg` (both teams mixed), `4227_now_teamB.jpg` (white team clean, 1 green),
`4227_now_teamA.jpg` (green team, still with the dimmest far whites and the spectators).

**Not fixed (-> P2d):**
- 13 of 40 whites still read green: the far, dimly lit whites have a torso colour (cream with green spill, L 45-60)
  nearer the green kit than the white one. Tried: local-light reading, mean/trimmed colour, far pair, green-kit mode,
  a classifier seeded from clear readings, letting the classifier decide only doubtful people - none better overall.
- Spectators by the right-hand fence (11 of 11) are read as the green team: they stand on the green run-off strip beyond
  the touchline, so both the grass test and the pitch-edge test keep them. Needs a touchline / side-edge cue.

**Free-runner re-track** (tools/p2c_track.py, same 20 s as P2/P2b): see below.
