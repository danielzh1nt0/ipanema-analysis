# N0-30b (30 Oct, worker, free runner + local, $0, nothing on Modal): AIK SECOND-half possession vs a blind by-eye sample

**Nightly card:** no 30 Oct card by 04:20 UTC -> backup nightly pushed (triggers/nightly.txt, free runner). The 29 Oct card says worse than 28 Oct = nothing.
**Queue before this:** M2 (main session) and PC2 (waits for M2) skipped, left open.

**Why:** N0-30a graded 60 first-half frames (27 decided, by eye 30% SFK vs the export's 39% - fits) and left the second half unchecked.
The Kaggle join (`results/kaggle/full_match_52d795-join`) says 45.5% SFK for the match and **51.7% in the second half**.

## How
- 90 random second-half times (3995-6899 s, seed 11) cut at full size, nothing drawn, on the free runner
  (`tools/n0_30b_aik_h2_cut.py`, video downloaded whole from R2 first, 4.0 GB, 90/90 frames) -> `results/free/n0_30b/h2`.
- Same grading as N0-30a, by eye, blind: full frame + 2.2x zoom around the exported ball spot (`tools/n0_30b_aik_h2.py views`).
  A = dark SFK, B = white AIK; restarts count for the team taking them (dead). Grades: `aik_h2_grades.json`; comparison: `result.json`.
- Teams/ends checked on the pictures: in the second half AIK defends the right-hand goal (red keeper); SFK's only goal (Veo 6268 s) is
  followed by the dark team walking back from the right box (q70) and a white kick-off (q71).

## Result
| | by eye: SFK share | 95% range | export at the same moments | export row agrees (nearest / +-1 s) |
|---|---|---|---|---|
| all 48 decided moments | **33%** (16/48) | 22-47% | 54% | **25/37** / 29/40 |
| 33 sure ones | 36% | 22-53% | 46% | 20/28 / 23/29 |
| 30 in play (restarts out) | 33% | 19-51% | 56% | 16/25 / 20/28 |
| **both halves pooled** (N0-30a + this) | **32%** (24/75) | **23-43%** | export whole match 45.5% | |

- The export's **second half (51.7% SFK) is outside the by-eye range** (22-47%), and so is its **whole match (45.5%)** against the pooled
  sample (23-43%). The first half fitted (N0-30a); the second half does not. By eye SFK had the ball about **a third** of the time in both halves.
- Moment by moment the export is right only 25/37 (68%) in the second half (first half 14/17). **10 of the 12 misses go the same way:
  AIK has the ball, the export says SFK.** Looking at the carrier the export picked (`misses_carriers.jpg`):
  - **AIK's red keeper is labelled SFK (team A, not a keeper)** every time he is the carrier (q17, q76; also in the labelled frame q56,
    3 of 3 times he is in the export on these pictures).
  - **White AIK players labelled SFK** (q31, q36): on 6 second-half frames with every exported player's team drawn (`labels/`), most are
    right, but in q64 (6050 s) 4-5 of ~9 white players are labelled A; q36 and q77 have one each. The export has more A than B players per
    frame in every second-half 5-min piece (A 6.8-9.0 / B 5.9-7.3) while the first half is the other way round (A 4.3-8.4 / B 6.3-8.8).
  - The referee usually gets a team label too (5 of 6 labelled frames) - a third source of wrong carriers near the ball.
- 42 of 90 frames can't be decided (ball in the air / not visible, set pieces forming, after the whistle).

## Side finding: the match ends ~2-4 min before the periods file says (not changed)
q81 (6777 s) onwards (9 frames) show the pitch after the final whistle: SFK in a huddle, spectators and children kicking balls; q80 (6679 s)
is still play. `periods/p15u-vs-aik-2026-09-21-bd09.json` ends the second half at 6899 s. The export gives SFK 64-66% of 6680-6899 s;
ending at 6680 s moves the second half 51.7 -> 50.8% (under 1 pt). Fix with the next AIK re-join, together with N0-30a's early starts
(first half 366 -> ~440 s, second ~3950 -> ~3998 s; q00 at 3997 s is still the kick-off being set up).

## What it means
AIK's whole-match possession (45.5% SFK in the Kaggle join) should **not** be shown: the blind sample says about 32% (23-43%), and the
gap comes from AIK players (the keeper above all) carrying SFK's label, not from the possession rule. Question for Daniel under
"Waiting for Daniel". Fix to try (new queue item N1c): the keeper team rule (a keeper takes the team of the goal he stands in, not his
shirt colour) and the referee filter on this ground - then re-grade on these 75 moments, free.

## Tests
`tests/test_n0_30b_aik_h2.py` (4): the sample times (seeded, second half after kick-off), the frame cutter on a stand-in video,
the grading view (zoom clamped inside the frame), and the committed grades (all 90 frames, blind, the numbers above).
Cutter dry-run locally on a stand-in video and through a local HTTP server before the runner push. $0, nothing on Modal.
