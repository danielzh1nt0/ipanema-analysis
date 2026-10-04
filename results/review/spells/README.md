# Are short possession spells real? (4 Oct, SFK-BP first half, by eye)

Daniel: "162 balls lost sounds extremely high". Export: 152 turnovers, 252 spells; 78 spells shorter than 3 s; 18 turnovers come
< 3 s after the previous one. Zoomed strips of 24 short (1.5-3 s) + 12 long spells: results/kaggle/spell_zoom (Kaggle, free).

First 11 short spells judged (5 frames each, 1.5 s before to 1.5 s after, our ball ring + player dots):

| spell | what happened | verdict |
|---|---|---|
| short00 | far-end duel, ring on the referee at start | unclear |
| short01 | B receives a long ball, keeps it (B>B>B) | real, not a turnover |
| short02 | black runs with the ball, white takes it | real turnover |
| short03 | black wins a tackle, white wins it back 2.7 s later | duel, real-ish |
| short04 | white dribbles, black tackles, ball loose, white again | duel, no real A possession |
| short05 | touchline, nobody at the ball at start | noise |
| short06 | throw-in by black, white takes it at once | real (lost throw-in) |
| short07 | far end, crowded, ball ends at the goal | unclear |
| short08 | B controls, ball goes out (B>B>B) | real, not a turnover |
| short09 | black has it ~2 s, white takes it | duel, real-ish |
| short10 | far-touchline duel, ring between players | noise |

Of the 9 that involve a team change: 2 clearly real, 2 duels where the team did briefly have the ball, 3 noise/duel without
real possession, 2 unclear. So roughly a third to a half of the SHORT spells are duels, not possessions.

What that means for the count: ~78 short spells, about half of them duels -> ~35-40 of the 152 turnovers are duel noise;
the real number is more like 110-120 per half (both teams), i.e. 55-60 per team. For reference Wyscout's "losses" for a
professional team run ~100-140 per 90 min = 50-70 per half, so our 73-79 per team per half is ~20-30% high, not 2x.

Decision (4 Oct): keep the counting rule (take 1.5 s) for the demo; the number is in a believable range once the duel share
is understood, and a 3-s rule would under-count real quick losses (lost throw-ins, tackles that stick). Label in the app:
"Losses (incl. duels) - Beta". After the demo: a duel is a loss only when the winner keeps the ball >= 3 s or plays a pass.
