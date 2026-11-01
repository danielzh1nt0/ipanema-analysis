# N0-01e (1 Nov, worker, free runner + local, $0, nothing on Modal): AIK - which way, and who scored? Kick-offs after the goals

1 Nov card: nothing worse than 31 Oct (tests 204/204). M2 (main session) and PC2 (waits for M2) skipped.

## Why
Open since N0-26b (Waiting for Daniel): in the AIK match the two half kick-offs say the dark team attacks LEFT in the
first half, but the periods file (4 Oct, from where the goals went in) says dark SFK attacks RIGHT. Both can't be true.
After every goal there is a kick-off too, and the laws fix two things there: each team stands in the half it defends,
and the team that **conceded** takes the kick-off. So the 7 Veo goals give 7 more checks.

## How
`tools/n0_01e_aik_kickoffs.py` on the free runner: 1280-px frames every 5 s from +25 to +90 s after each Veo goal clip
(2762, 3128, 4257, 4962, 5625, 5916, 6268 s) and around the 84' penalty -> `results/free/n0_01e/` (109 frames, sheets
per goal). Centre crops of the decisive frames (`tools/n0_01e_crop.py`): `k<goal>.jpg` here; `ko1.jpg` = first-half
kick-off (from N0-31d).

## What the pictures show
| moment | line-up | at the ball | where play was just before |
|---|---|---|---|
| first-half kick-off 445-465 s | white LEFT half, dark RIGHT | dark | - |
| goal 2762 | players still walking back at 2812-2822, mixed | can't tell | LEFT end (keeper in red) |
| goal 3128 | mixed at 3162-3170 | can't tell (dark + white near the ball) | LEFT end (3128-3152) |
| second-half kick-off 3995-4015 (N0-31d) | dark LEFT, white RIGHT | white | - |
| goal 4257, frames 4292-4302 | **dark LEFT, white RIGHT, clean** | **white** | - |
| goal 4962, 5012-5022 | mostly dark left / white right, not clean | can't tell | RIGHT end (4987-5002) |
| goal 5625, 5660-5670 | darks walking back from the right end | can't tell | RIGHT end (5650) |
| goal 5916, 5956-5966 | dark left, whites at the centre spot | **white** | (5941: play at the left end - maybe already after a kick-off before +25 s) |
| goal 6268 (Veo: 84' AIK penalty, then "SFK Kickoff"), 6318-6328 | darks walking back from the right end | **white** | RIGHT end, penalty-box crowd with a yellow keeper |

All clean line-ups agree with each other: **the dark team defends the right half in the first half (attacks left) and
the left half in the second (attacks right).** All placed goals fit **the dark team scoring**: both first-half goals at
the left end (where dark attacks), 4962 / 5625 / 6268 at the right end in the second half (where dark attacks), and the
white team kicks off after 4257, 5916 and 6268. The final score has one goal for the other side;
which one is not clear (5916 is the odd one: play at the left end at 5941, yet white at the centre spot at 5961), but no
goal fits the file's version (white attacking left in the first half).

## So which club is dark?
- Veo's own labels: "84' AIK penalty, AIK goal 1-6, 85' SFK kick-off" - and the white team takes that kick-off. So
  **Veo says dark = AIK**, which makes everything above consistent with the 1-6 AIK win and the 4 Oct "AIK 2-0 at
  half-time".
- Our pipeline and the 4 Oct note assume **dark = SFK** (SFK is dark at BP and Vallentuna; the white team here wears
  black-and-yellow hooped socks, AIK's colours).

If dark = SFK, then SFK won 6-1 (Veo's team labels are swapped) and our file has the direction wrong.
If dark = AIK, then the score 1-6 is right, the direction file is right for SFK, and our export has the TEAMS swapped
(SFK's label sits on AIK's dark kit).

**Either way the app's AIK match is wrong today**: it has the dark team (A) attacking right in the first half and the
goals going to the white team (B). The pictures say the dark team attacked left and scored. Only Daniel knows which
club wore dark -> Waiting for Daniel (one question: "Did SFK beat AIK 6-1, or lose 1-6?").

What it would change (nothing changed yet):
- SFK won 6-1 (dark = SFK): `attack_right_A` -> false in `periods/p15u-vs-aik-2026-09-21-bd09.json`, Veo goals of this
  match -> team A; mirrors field tilt / thirds / forward passes / high turnovers / goal ends for AIK.
- SFK lost 1-6 (dark = AIK): swap A/B for the AIK match (team names, possession %, every per-team number);
  direction and goal teams stay. Also the by-eye possession sample (N0-30a/b, "SFK ~32%") then means AIK ~32%.

## Tests
`tests/test_n0_01e_aik_kickoffs.py` (3): windows follow the Veo goal list, every kick-off window long enough, dry run on a
stand-in video (1280 px out). Dry run before the push. Full local suite 795 pass (only the known `supervision` module
missing here). Free runner only, $0, nothing on Modal.
