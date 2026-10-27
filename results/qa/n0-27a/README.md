# N0-27a — "A better pass was open" (27 Oct, worker, local $0)

Nightly card: no 27 Oct card yet (00:20 UTC, before 03:00); 26 Oct card says nothing worse than 25 Oct.

**What it is:** `stats.json teams[].better_option_count` (Insights card "A better pass was open"), `players[].better_option_count`,
`passes[].better_option / best_to`, events of type `better_option` (already tier hidden). Never checked before.
Rule (`analytics.best_option`, pulled out of `passes()` unchanged - same output on a Metrica half, md5 identical):
each team-mate in reach (45 m) is valued gain towards goal + 4 per opponent "bypassed" + 0.5 per metre of space (max 10),
-25 if an opponent stands within 1.5 m of the lane; "better" = another mate valued >= 6 above the one played to.

## 1. The rule itself, perfect inputs (Metrica pro data, all 22 players + the ball) - `tools/n0_27a_rule.py` -> `metrica_rule.json`

| rule | completed pro passes called second best | cut-out (intercepted) passes called second best |
|---|---|---|
| app | **83%** (1,439 / 1,726) | **66%** (168 / 256) |
| margin 12 | 75% | 55% |
| better option within 30 m | 74% | 52% |
| lane widens 0.1 m per metre | 74% | 63% |
| widen + 30 m | 66% | 50% |
| widen + 30 m + margin 12 | 53% | 39% |

Backwards on every variant: a pass that worked is called a bad choice MORE often than one that was cut out. Same on each of the 4 halves.
Why: the "better" option is the furthest-forward mate (median 18 m forward, 61% >= 15 m, 99% "bypass" >= 1 opponent; played value median 3, best 35).
By eye (`sheet_metrica.png`, 12 random flagged completed pro passes): the "better" mate is mostly among or right next to opponents
(~7 clearly not better, ~1 plausible switch of play, ~4 can't tell).

## 2. Our inputs vs the rule on perfect inputs - `tools/n0_27a_better_option.py` -> `metrica_better.json`

| our inputs (Metrica) | flagged of valued passes | count vs perfect | per-pass agreement | same "better" mate |
|---|---|---|---|---|
| perfect ball, all players | 83% | 0.92-1.16x | 94% | 87% |
| flicker ball (like our clips) | 82% | 0.88-1.30x | 91% | 83% |
| our camera views (~40% of players seen) | 63% | 0.23-0.49x | 71% | 36% |
| camera views + flicker | 65% | 0.36-0.65x | 72% | 35% |

The pipeline reproduces the rule well; through our camera the "better" mate is usually someone else (the real one is out of the picture).
App copies: 70-78% of valued passes flagged (SFK-BP 80/104, AIK 89/127, Vallentuna 246/316).

## Result

The number is not a decision-quality measure: it says nearly every pass was second best, and more so for passes that worked.
No variant tried fixes that (it needs a pass-risk / receiver-pressure model, not a tweak). Built `IPANEMA_BETTER=withhold`
(teams[] and players[] better_option_count = null, app shows Withheld), default unchanged -> Waiting for Daniel.
tests/test_n0_27a_better.py (6).
