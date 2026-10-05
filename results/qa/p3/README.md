# P3 - joining broken player tracks (5 Oct, worker, local + free GitHub runner, $0, no Modal)

**Question.** A player's track breaks into pieces. Can we join the pieces afterwards (the gta-link idea: by place, time and
look, never two pieces on screen at the same time)?

**Pieces checked** (20 s each): Reymersholm 726 / 2227 / 4227 and Spånga 1159 / 2576 / 3994 (latest free-runner
tracking, screen positions, no calibration), SFK-BP from the app's full-match export at 900 / 1800 / 2700 s (metres; the
app already joins pieces < 2.5 s apart).

## Today (before joining)

| piece | track pieces per 20 s | median s a piece lasts |
|---|---|---|
| Reymersholm 726 / 2227 / 4227 | 27 / 37 / 25 | 7.3 / 8.5 / 3.7 |
| Spånga 1159 / 2576 / 3994 | 37 / 57 / 47 | 9.6 / 2.4 / 4.3 |
| SFK-BP app 900 / 1800 / 2700 s | 38 / 29 / 42 | 5.2 / 7.6 / 1.6 |

About 13-17 people are in view, so a player is cut into roughly 2-3 pieces per 20 s.

## Joining, graded by eye

Tool: `ipanema/tracklets.py` (option only, not in the pipeline), `tools/p3lab.py`. Crops of every candidate piece were
cut on the free runner (`tools/p3_crops.py`, no detector), each proposed join shown as "end of piece A | start of piece B"
in `pairs/*.jpg` and graded by eye (`grades.json`, scored by `tools/p3_score.py`).

| rule | pieces per 20 s | joins: same player / different person / both not players / can't tell |
|---|---|---|
| place + time, gaps up to 5 s | -10 to -36% | **19 / 27 / 13 / 28** |
| same + shirt/shorts colour veto | -10 to -35% | 18 / 17 / 11 / 26 |
| only gaps <= 1.5 s and <= 3 m | -0 to -16% | 10 / 2 / 3 / 5 |

**Result: no gain, nothing switched on.** Joining by place and time makes more wrong joins than right ones. The safe short
rule is mostly right but only removes a few pieces (and the app already does almost the same for gaps < 2.5 s).

## Why

- **Many "pieces" are not players**: bench people, spectators behind the fence, a bin, the goal frame, an ad board
  (13 joins between two non-players; several wrong joins link a player to the bench). On Spånga 2576 most candidate
  pieces are spectators behind the fence. Dropping off-pitch tracks first would cut the piece count more than any joining.
- **Colour can't tell team-mates apart**: look distance between two halves of the same track (median 0.07) overlaps with
  two different players of the same team (median 0.17). The colour veto mainly removes kit/referee/bench mix-ups.
- Follow-cam pans and zooms make long gaps (> 2 s) hard to bridge by position; long gaps need a real look model
  (person re-identification or shirt numbers) - that is M2's job.

Takeaway for M2: (1) remove off-pitch people before joining, (2) join short gaps by place, (3) long gaps only with a
person look model or shirt numbers - colour histograms are not enough.

Files: `p3lab.json` / `lab_pos.json` (place + time), `lab_look.json` (colour veto 0.25), `lab_short.json` (1.5 s / 3 m),
`emb.json`, `crops/`, `pairs/`, `grades.json`. Test: `tests/test_p3_tracklets.py`.
