# F1 — full matches vs the 5-min clips (2 Oct)

## SFK-BP ball: 25/34 in the full match vs 29/34 in the clip — found

The clip starts at 1200 s = the first frame of full-match piece 4, so the full run sees the same frames, the same
line calibration and the same ball finders. The pictures line up exactly (`results/modal/SFKBP1109_full_ballcheck.jpg`
vs the clip's sheet): the green answer circle sits on the ball in both, so the frame mapping is right.

The difference is one rule the full run adds: frames where the line calibration is unsure (13% of frames, bridged or
borrowed camera) are blanked — no players and **no ball guesses**. The clip keeps them (it only stops drawing pitch lines).
3 of the 34 checked moments (clip frames 1383, 2536, 4150) are on such frames; on the full sheet they are exactly the 3
"miss" tiles with no guesses at all. The 4th lost moment (39883 / clip 3919, picked 148 px off) comes from other
full-match differences (players / kits), not reproduced offline.

Replayed offline on the clip's exact app inputs (`tools/f1_sim.py` -> `sim.json`):

| | 34 moments | ceiling | B4 key (284) | who has the ball (99) |
|---|---|---|---|---|
| clip (nothing blanked) | 29 | 34 | 284 | 74 |
| full match until now (players + ball guesses blanked) | 26 | 31 | 240 | 74 |
| ball guesses kept, players blanked | 29 | 34 | 284 | 70 |

**Fix (code, default):** for line-calibrated pieces the full run now keeps those frames the way the clip does — players and
ball kept, frame marked "unsure" in the export (no pitch lines drawn). `IPANEMA_FULL_UNSURE=blank` gives the old behaviour.
(`ipanema/fullmatch.py trusted_frames`, `modal_app.run_full`; test `tests/test_f1_unsure.py`.) Expected full SFK-BP: back to the
clip's 29/34 on these moments, B4 284. Reaches the app only with a re-run of the full SFK-BP (pieces cached, CPU) -> Daniel's go.

## AIK players 6+6 vs 7+7 and ball 26/39 vs 30/39 — being checked

- Full AIK removed 3.8 "not a team kit" people per frame vs 1.0 in the clip (794,212 vs 9,086 rows). SFK-BP shows no such gap
  (2.0 full vs 3.0 clip). The full run learns one kit model for the whole match; for AIK it ran before the halves were known, so
  it sampled the default spans. Free-runner check started: `tools/f1_check.py` -> `results/qa/f1/` (the three kit models on the
  same people + pictures, and 10 app moments of each full match).
- Ball: AIK has no unsure blanking (trust_all), ceiling 39/39 in both. Lost in the full run: 79495, 79945, 80094, 82192, 84140
  (clip 3971, 4421, 4570, 6668, 8616); gained 78146 (clip 2623). Each full piece has its own panorama registration and the
  full run's players differ, so the picker's metres/near-player terms differ. Proof needs the full run's picker inputs
  (`cache/p15u-vs-aik-2026-09-21-bd09/picker_inputs.pkl` on Modal) -> waiting for Daniel (a Modal fetch).
