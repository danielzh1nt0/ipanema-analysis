# N0-30d (30 Oct, worker, local + free runner, $0, nothing on Modal): 'X m in 5 s' after winning the ball

**Card:** the 30 Oct card's one 'worse' row (tests failing: test_n0_30b_aik_h2) was already fixed this morning (N0-30c,
commit 77977cf; passes here). M2 / PC2 skipped (main session / waits for M2).

**What was checked:** every 'won the ball' event in the app says "forward pass … s · **X m in 5 s**"
(match_data events `turnover_won`, `payload.gain_5s_m`, from `P.turnovers`): how far the ball moved towards the winner's
attacking goal in the 5 s after the win. Never checked before.

## Fault found
It read the ball at exactly +5 s, wherever it projected. On the repo app copies: SFK-BP full **923 m** and **-406 m**
(pitch is 106 m), Vallentuna -152 m (rebuild); about 1 win in 20 beyond 40 m. Cause: at +5 s the ball pick is in the air
(projects tens of metres behind a goal), out of play by the fence / bench, on the spare ball at the post, or on a frame
with an unsure camera.

## Fix (default, `IPANEMA_GAIN_ONPITCH=0` = old)
`P.gain_window`: both ends must be balls on the pitch (+2 m). Start = first such ball in the first second; end = the
LAST such ball within the 5 s (a ball that goes out counts to where it left); under 1 s of usable ball -> no figure ('—').
`run.py` and `tools/metricalab.py` pass the pitch size; other callers keep the old rule.

## Metrica pro data (tools/n0_30d_gain5s.py, results/qa/n0-30d/metrica_gain5s.json)
Truth = the true ball 5 s after each labelled win (BALL LOST -> RECOVERY), matched one by one to our kept turnovers.

| ball | rule | wins with a figure | error (mean / median) | within 5 m | forward/level/back right | 'always 0 m' error |
|---|---|---|---|---|---|---|
| perfect | old | 82% | 4.2 / 1.3 m | 72% | 85% | 14.7 m |
| perfect | **new** | **95%** | **3.9 / 1.2 m** | 74% | 85% | 14.1 m |
| flicker (our clips' kind) | old | 82% | 5.7 / 3.5 m | 63% | 80% | 15.5 m |
| flicker | **new** | **92%** | 5.8 / 3.4 m | 63% | 80% | 15.4 m |
| streak | old / new | 69% / 88% | 5.6 / 5.7 m | 59 / 60% | 72 / 74% | ~12 m |

The figure itself holds (3-4x better than guessing 0); most of the error is WHEN we call the win (true ball from our
win frame: 3.9-5.5 m), not ball noise (1.4-2.2 m). Metrica has no wild projections, so the new rule's gain there is
coverage only (it blanks the ball when it is out).

## Our app copies (tools/n0_30d_clips.py, rebuilt from the exported frames; results/qa/n0-30d/clips_gain5s.json)
| match | wins | rebuild = stored (within 2 m) | over 40 m old -> new | largest old -> new | changed > 3 m |
|---|---|---|---|---|---|
| SFK-BP full | 146 | 118/145 (copy mixes older exports) | 11 -> 4 | 406 -> 51 m | 29 |
| Vallentuna | 270 | 238/270 | 10 -> 8 | 153 -> 53 m | 22 |
| AIK (2 of 12 frame files in the repo) | 45 rebuilt | 39/45 | 1 -> 1 | 57 -> 57 m | 0 |

## By eye (free runner, tools/n0_30d_frames.py -> results/free/n0_30d/g00-g21.jpg; grades.json)
All 22 changed Vallentuna wins: crops around the ball pick at the start and at both end points, ends shown as X / Y in
a seeded random order, graded before reading the key. **New end better 8, old better 1, same 7, can't tell 6.** The new
rule's wins: old end on the spare ball at the post (2), the ball in the air over the goal (old -153 m, new -32 m), the
bench / fence / an empty spot (5). Old's win: new end on empty grass while the ball was at a player's feet (g17).

## Not changed
- Remaining big on-pitch values (Vallentuna 8 over 40 m) are mostly real long balls or the picker jumping; not touched.
- On our clips the average gain is ~0-5 m vs +9 m on pro data; could be the ball, the win timing or youth play - not
  investigated.
- Reaches the app at the next run / re-join (Modal, Daniel's go).

## Tests
`tests/test_n0_30d_gain5s.py` (9): same as old on the pitch, wild end ignored, ball out over the end line counts to the
line, start off the pitch, < 1 s usable -> None, no pitch -> old rule, turnovers use the pitch (and the env switch), the
lab truth + score, blind picks reproducible. Full suite 813 pass here (4 need supervision, not on this box).
Free-runner job dry-run first on a local stand-in video.
