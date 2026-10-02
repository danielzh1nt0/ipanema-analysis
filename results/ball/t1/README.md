# T1 - second opinion on each ball guess before the picker (2 Oct)

**Result: no gain, nothing switched on.** The small 3-frame scorer cannot tell the ball from a shoe.

## What was done ($0, no Modal)
- Trained 4 small 3-frame ball-vs-not scorers on this machine (`tools/t1_train.py`): SFK-BP clicks only, and SFK-BP + 4 other matches (K1 crops), 2 seeds each. Crops from inside the SFK-BP test clip were left out, and AIK was never used, so both tests are fair.
- Free GitHub runner (`tools/t1_score.py`, 6 min): scored every guess the app's picker got on both clips (AIK 128k guesses, SFK-BP 149k) -> `results/free/t1/`.
- Here (`tools/t1lab.py`): removed or lowered the doubted guesses, re-ran today's picker on the exact app inputs, graded on the keys. -> `t1lab.json`

## Numbers (picker within 30 px of the ball)
| | AIK /39 | SFK-BP /34 | B4 key /285 |
|---|---|---|---|
| app picker today | 30 | 29 | 284 |
| best: drop guesses the scorers are sure are not the ball (p < 0.05) | 30 | 30 | 284 |
| drop p < 0.2 | 26-29 | 29-30 | 279-284 |
| soft lowering (any strength) | 21-28 | 26-30 | 277-285 |

The only gain is 1 SFK-BP moment; AIK never goes up with a fair scorer, so by the rule it is not kept. (The older Kaggle scorer reaches AIK 31 once, but it saw SFK-BP clicks from the test clip and loses 7 B4 moments.)

## Pictures (`results/free/t1/sheet_*.jpg`: every guess at each key moment, scorer's number, green = the real ball)
- The real ball usually gets 0.8-1.0, but so do many shoes, legs, line bits and empty grass (0.9+ is common). It does not separate ball from shoe, which is exactly the picker's remaining mistake (results/qa/ballmiss2).
- At most key moments the finder's top guess is already the ball; the scorer mostly adds noise there.

## Next
A fourth try at this scorer is not worth it. The shoe/ball problem needs better ball pictures in the finder itself (S3 pasted balls) or the picker using movement over time (PF1).
