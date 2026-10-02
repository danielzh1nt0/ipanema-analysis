# F1c — kit colours in sun and hard shadow (2 Oct, free, $0, no Modal)

Problem (F1b): on the full AIK match one kit model mixes the white and the dark team, because half the pitch is in hard
shadow: a white shirt in shadow and a dark shirt in sun end up about the same lightness.

Fix (`ipanema/kits.py`): read each player's shirt lightness **relative to the grass at his feet** (`local_grass_L`,
grass found by its hue, so shaded grass counts too). A white shirt is "much lighter than the grass where he stands"
in sun and in shade alike. Used only when a match really has sun + shade: `shade_spread` (how much the grass at the
players' feet varies, p90/p10) must be >= 2.4. Env `IPANEMA_KIT_LIGHT=frame` = old reading, `local` = always.

| ground | shade spread | reading chosen | players in the right team (by eye key) |
|---|---|---|---|
| **AIK full match** (30 test frames, 254 players, new key) | 2.85 | local | **130 -> 234 / 254**; wrong team 29 -> 3; players called "neither" 95 -> 17 |
| Spånga (night) | 1.96 | frame (unchanged) | 233 / 279 (local would give 229) |
| Reymersholm (night) | 1.54 | frame (unchanged) | 239 / 252 (local would give 231) |
| SFK-BP clip / full match | 1.15 / 1.23 | frame (unchanged) | no change (0 of 516 people) |

AIK per frame on the pitch: dark 5.1 -> 6.0, white 3.6 -> 5.6, neither 4.8 -> 1.9. Others (referees, keepers,
spectators) kept out of the teams: 39 -> 34 of 120 (a little worse).

By eye (`p15u-vs-aik-2026-09-21-bd09_strips.jpg`): the old model's dark team (top) holds many whites; with local light
the dark team is dark players and the white team is white players, a few dark players in deep sun still in the white
strip. `_changed_BtoA.jpg`: almost all are dark players moving back to the dark team. `_into_neither.jpg`: mostly
referees, keepers, spectators, plus some players.

Also seen: on AIK the "feet on grass" test used for kit learning keeps mostly people in the shadow (the frame's grass
reference is the shaded lower part of the picture), so the gate is measured on everyone, before that test.

Caveats: the gate is set from one sunny match (AIK) and four even-light/night ones; the AIK key is Claude's eye on
blurred crops (31 marked unclear, left out). Reaches the app only at the next full run (Modal: waits for Daniel).

Files: `aik_labels.json` (405 people: D 132, W 122, O 120, U 31), `f1clab.json`, sheets `*.jpg`, frames in `frames/`
(36 fit + 30 test AIK, 36 + 18 SFK-BP, from the free runner: `tools/f1c_frames.py`). Re-run: `python tools/f1clab.py`.
Test: `tests/test_f1c_local_light.py`.
