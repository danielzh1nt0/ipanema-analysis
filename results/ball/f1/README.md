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

## AIK players 6+6 vs 7+7 — main cause found: the one kit model for the whole match

- Full AIK removed 3.8 "not a team kit" people per frame vs 1.0 in the clip (794,212 vs 9,086 rows); SFK-BP has no such gap
  (2.0 full vs 3.0 clip). The full run labels teams with ONE kit model learned from 36 frames spread over the whole match
  (modal_app.rf_kits); the clip learned its own from 36 frames of its 5 minutes.
- Free-runner check (`tools/f1_check.py` -> `results/qa/f1/summary.json`, `aik_kits_match_vs_clip.jpg`): the three models
  rebuilt from the same frames label the same people (feet on the pitch). Per frame, inside the clip window:
  match model as used dark 3.5 / light 2.9 / **neither 2.8**; clip model 4.3 / 3.4 / **1.4**. Over both halves: neither 3.2 vs 2.1.
  A match model from the halves only (periods file; 1 of the 36 default frames was in half-time) is no better (2.2 / 3.3).
- By eye (6 moments): the match model turns real players into "neither" — 3 far dark players at 2748 s, white #9 at 2592 s
  (the clip model gets these right). The clip model is not perfect either: white #9 in the shadow at 2800 s read dark.
  Sunlight and long shadows change the kit colours over 2 hours; one model for the whole match fits worse than one per 5 min.
- Next (proposal, queue F1b): a kit model per piece, with each piece's two teams matched to the match model's teams by colour
  so A/B can't swap between pieces. Needs the players re-tracked (GPU; Kaggle is free).
- The app-view pictures (part B) could not be made: Supabase storage answers 400 to public reads (bucket not public).

## AIK ball 26/39 vs 30/39 — not proven yet

AIK has no unsure blanking (trust_all), ceiling 39/39 in both. Lost in the full run: 79495, 79945, 80094, 82192, 84140
(clip 3971, 4421, 4570, 6668, 8616); gained 78146 (clip 2623). Each full piece has its own panorama registration and the
full run has fewer players (above), so the picker's metres and near-player terms differ. Proof needs the full run's picker
inputs (`cache/p15u-vs-aik-2026-09-21-bd09/picker_inputs.pkl` on Modal) -> waiting for Daniel (a Modal fetch).
