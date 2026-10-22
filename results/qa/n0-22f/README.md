# N0-22f (22 Oct, worker, local only, $0, nothing on Modal)

**Card 22 Oct:** worse than 21 Oct = nothing (tests 152/152, ball 22/34, pro-data passes 15% / possession 8 pts off).
M2 / PC2 / BALL-EDGE skipped (main session / waits for M2 / needs a Kaggle GPU; this worker runs only local + the free runner).

**What was looked at:** the counter-press figure in `stats.json` teams[]: `regained_within_5s_pct` ("Back in 5 s" in the
app, the base of its pressing verdict) and its mirror `lost_back_5s_pct`. Nobody had checked it (accuracy status 17 Oct:
"pressing ... not checked by anyone yet").

## How it was checked (no clicking)
- Truth = Metrica's free pro games (2 games x 2 halves): after each labelled BALL LOST, does that team start its next
  on-ball chain within 5 s? (`tools/n0_22f_regain.py`, `truth_regains`)
- Ours = the app's own steps run on Metrica's positions (`tools/metricalab.ours`: possession state -> spell state ->
  turnovers -> S9 duel rule), with a perfect ball and with our 'streak' ball noise.

## Finding
| half | truth | app rule, perfect ball | app rule, our streak noise |
|---|---|---|---|
| game 1, 1st | 28% of 142 losses | 2% of 49 | 4% of 23 |
| game 1, 2nd | 22% of 115 | 4% of 57 | 4% of 23 |
| game 2, 1st | 20% of 128 | 14% of 66 | 11% of 27 |
| game 2, 2nd | 17% of 105 | 19% of 52 | 0% of 22 |

Picture: `regain_vs_truth.png`. Full rows: `metrica_regain.json`.

- Why: a ball won back quickly is mostly **not a counted loss at all**. The counted state joins a team's spells across
  short gaps (take 1.5 s / join 3 s) and the S9 rule drops losses where the winner keeps it < 3 s without a pass. So the
  losses that are left are the ones the other team kept - and those are rarely won back. Same population in the truth
  (S9-style losses only): 7-10% regained.
- On the losses we do find (matched to the truth within 2 s), our yes/no agrees 80-100% of the time, but mostly on
  "no"; of 5-8 true regains per half (perfect ball) the app rule catches 3-5.
- Tried instead (`spell_sweep.json`, `raw_sweep.json`):
  - straight from the counted state: 4-10% with a perfect ball, 0-3% with streak noise.
  - from the RAW state, dropping spells < 0.3 s: **19-29% with a perfect ball - on the truth** - but 7-12% with streak
    noise and **79-84% with the flicker noise** that looks like our clips (ball dot hops between nearby players).
  - Nothing tracks the truth under our noise. The repo copies of our matches show 11-29% (full halves) and 78-82% (the
    SFK-BP clip) - the clip's number is the flicker signature.

## Change (default unchanged)
- New option `IPANEMA_REGAIN=withhold`: both team figures become `null` plus `regain_withheld` (the app then shows a dash,
  stat register #1). Default `on` = today's numbers. Per-event `regained_within_5s` in the turnover events is untouched.
- Not switched on by the worker: hiding a headline app figure is Daniel's call -> Waiting for Daniel.
- `tests/test_regain.py` (3: default unchanged, withhold, the truth reader on hand-made events).

## Not changed / notes
- "Pressed within 2 s" / time to press: Metrica has no pressure labels, so not checkable this way.
- Locally, with `ultralytics` installed, `tests/test_tracking_batch.py` fails (another test leaves a stand-in
  `ultralytics` module behind); same on main before this change; the nightly runner has no ultralytics, so it skips there.
