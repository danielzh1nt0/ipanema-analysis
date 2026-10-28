# N0-28b (28 Oct, worker, local + free runner $0): sequence length and metres gained

Card: no 28 Oct card at 04:20 UTC -> backup nightly pushed; the 28 Oct card then came in: nothing worse than 27 Oct.
M2 (main session) and PC2 (waits for M2) skipped.

## What was checked
`stats.sequences[].duration_s` and `.gain_m` - on screen as each 'sequence ended' event ("12.4 s · gain 18.0 m") and
the Ball tab's 'How long did we keep it?' card. Never checked against a truth (S6 / N0-25e did the count, reach and end labels).

Metrica pro data, 2 games, 4 halves, perfect ball + our 'streak' / 'flicker' ball noise, 2 seeds (`tools/n0_28b_seqlen.py`,
table `tools/n0_28b_summary.py` -> `summary.txt`). Truth = chains of one team's on-ball events (N0-25e rule); 'poss.' =
the same with an opponent touch under 1 s not ending the chain.

## Result (old rule, today's app)

| ball | count ours / truth | median length ours / truth (poss.) | 90th pct ours / truth | matched within 2 s | metres gained: median error, direction right |
|---|---|---|---|---|---|
| perfect | 473 / 751 | 9.6 / 3.7 (5.4) s | 41 / 23 s | 44% | 10.3 m, 84% |
| streak | 1548 / 1502 | 2.5 / 3.7 s | 10 / 23 s | 30% | 14.1 m, 64% |
| flicker (like our clips) | 1044 / 1502 | 9.4 / 3.7 s | 38 / 23 s | 39% | 10.8 m, 81% |

- **Lengths read about 2x long** with a perfect or flickering ball (our clips: SFK-BP median 5.4 s, AIK 7.0 s, longest 32-42 s).
- Cause: our sequences run on through stoppages. 12-15% of them contain a real stoppage (`across_stops.json`), and those
  hold 19-41% of all sequence time (median 23-46 s vs 5-9 s for the rest). Most are **free kicks after a foul taken by
  the same team** (31 of 42 with a perfect ball, `inside_restarts.json`) - the ball never goes out, so nothing in our
  inputs says play stopped. Swallowing a real opponent spell of 2 s+ is rare (2% perfect, 5% flicker).
- Metres gained: right direction ~8 in 10, but a median 10 m off (a long sequence's end is far from the true end).
- Streak noise cuts sequences too short instead (2.5 s).

## Tried, not switched on (options)
- `IPANEMA_SEQ_SPLIT_STOP=1`: cut a sequence at each stoppage the restart step finds (`P.split_at_stops`). Metrica:
  small gain with flicker (median 9.4 -> 8.2 s, within 2 s 39 -> 43%, gain error 10.8 -> 10.3 m), neutral otherwise -
  the restart step sees few of the real stoppages. App clips (`clips.json`): SFK-BP 32 -> 28 sequences (13 change),
  AIK 1 change. **Blind by eye** (16 sheets, 8 of our stoppages + 8 control windows, `grades.json`, key unsealed after
  grading): of our 8 stoppages, 3 stopped or a restart, 2 clearly play on, 1 play on then unclear, 2 can't tell; of the
  8 controls 1 was a real stoppage we miss. Cutting there would cut real play about as often as it helps -> off.
- `IPANEMA_SEQ_GAIN=edge0.5`: metres gained from the median ball of the first / last 0.5 s instead of one frame:
  missing values 5.8 -> 1.8% on Metrica, error unchanged; our clips have no missing values (bridged ball) -> off.

Nothing in the app changes. Question under Waiting for Daniel (label the lengths "rough" or hide the length card).
Tests: `tests/test_n0_28b_seqsplit.py` (6). Suite 715 pass (+ 7 skipped). $0, nothing on Modal.
