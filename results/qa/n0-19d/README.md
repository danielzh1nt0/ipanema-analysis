# N0-19d (19 Oct 12:20 UTC, worker, local, $0, nothing on Modal)

**Nightly card 19 Oct:** worse than 18 Oct = nothing (tests 134/134, ball 22/34, passes 15% / possession 8 pts off on pro data, all 7 player rows run). No new main-session code since 08:20, so I tested the parts of `ipanema/analytics.py` that every app export runs and nothing tested: passing lanes and team shapes.

## Finding: the export says the kits the wrong way round
- Every `match_data.json` says `"teams": {"light": "A", "dark": "B"}`. The pipeline names the **darker** kit "A" (kits.py, teams.py, tracking.py; "SFK is team A, the darker kit" in all three app matches). So the label was backwards in every export.
- Proof on the app's SFK-BP clip (repo copy), scored against the by-eye who-has-the-ball key that is written in kit colours (`tools/team_letter_check.py`, output `team_letter_check.txt`): reading A as dark gives **33 right / 6 wrong**, reading A as light **6 / 33** (58 clear moments).
- By eye on `results/qa/tracktest/compare_0256.jpg`: the "before (app, 27 Sep)" panel puts the "dark" circles on the white players. `tools/tracktest.py` took the dark team from that label, so its "before" column has always had dark and light swapped (it says dark 10 / light 4; really dark 4 / light 10). The "new" column - the one the nightly card reports - was right.
- In the repo only tracktest read the label. I could not see whether the Lovable app reads it; if it colours teams from `teams.light/dark`, they were swapped there too.

## Fix
- `ipanema/export.py`: writes `"teams": {"dark": "A", "light": "B"}` (`EX.TEAM_KITS`). Reaches the app at the next run.
- `tools/tracktest.py`: the old app export's dark team is "A" (no longer read from the wrong label).
- Tests: `tests/test_team_letters.py` (4: constant, export writes it, scorer, the repo SFK-BP clip says A = dark). Local suite 460 pass, 6 skipped (no torch here).

## Lanes and shapes: no fault found
`tests/test_lanes_shapes.py` (6): open vs blocked lane, forward follows the attack direction, opponents past the receiver or 3 m off the line don't block, far team-mates (> 45 m) left out, shape leaves out keepers and needs 3 players, a shape is held 1.5 s when players drop out, then cleared.

## Seen, not changed
On 20% of frames with a carrier the export's `possession` names the other team (Vallentuna 3,010 of 15,508, AIK 344 of 1,603, SFK-BP clip 370 of 2,693). `possession` is the graded state (possession_simple), `carrier` is just the nearest player within 2.5 m (often an opponent in a duel); the lanes are drawn from that carrier. On the 4 key moments where they disagree each is right twice, so neither should replace the other. If the app shows lanes for the team not in possession, the app could show lanes only when the carrier's team = `possession`.
