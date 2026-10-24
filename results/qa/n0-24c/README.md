# N0-24c (24 Oct, worker, local + free runner $0): final-third entries on pro data

Card: 24 Oct card says nothing worse than 23 Oct. M2 (main session) and PC2 (waits for M2) skipped.

## What was checked

The app's final-third entries (Shooting tab, `metrics.field[tm].entries_count` + the list), never checked on pro data
before (N0-20b only by eye). On Metrica's 2 pro games, 4 halves (`tools/n0_24c_entries.py` -> `metrica_entries.json`,
summary `tools/n0_24c_summary.py`). Truth = Metrica's events + clean ball: the ball crosses the 2/3 line while the team
owns it (a pass in flight belongs to the passer's team), same re-arm as the app. Ours = the pipeline on Metrica's
positions, perfect ball and our two ball-noise models (2 seeds). Matched within 2 s, same team.

## What was wrong

Our possession state calls a pass in flight 'loose', and the entry rule needed control within 3 frames on both sides
of the line. So **a pass into the final third was almost never counted**: with a perfect ball we found 42% of the
entries, and only 19% of those made by a pass. The count read about half the truth.

## Fix (on by default, IPANEMA_ENTRY_HOLD_S=0 = old)

The team keeps the ball across loose frames for up to 3 s; a crossing across the other team's control never counts;
an entry across a gap is labelled 'pass'. Totals over 4 halves (noise rows = 2 seeds, so twice the truth):

| ball | rule | entries found | ours real | count vs truth | pass entries found |
|---|---|---|---|---|---|
| perfect | old | 42% | 91% | 0.46x | 19% |
| perfect | **new** | **72%** | **96%** | **0.75x** | **85%** |
| streak noise | old | 14% | 62% | 0.23x | 6% |
| streak noise | **new** | **31%** | **78%** | **0.40x** | 25% |
| flicker (like our clips) | old | 51% | 54% | 0.94x | 32% |
| flicker | **new** | **65%** | **62%** | **1.04x** | 58% |

Better on every row (1 s and 2 s holds sit in between, `metrica_entries.json`). A first lab version that also re-armed
on any touch by the other team made 18% too many under flicker (`metrica_entries_lab_v1.json`); not used.
Pass vs carry label right 66% (was 60%) under flicker. Field tilt untouched.

## App clips (exact inputs, `tools/n0_24c_clips.py` -> `clips.json`)

SFK-BP clip: dark 11 -> 10, light 9 -> 8. AIK clip: dark 0 -> 0, light 32 -> 33. Only 5 entries change.
By eye (free runner sheets `results/free/n0-24c`, ball ring red, 2/3 line yellow; not blind - I wrote the key):

- dropped (3): sfk00 15.4 s and sfk03 232.1 s = the ball pick on the touchline clutter spot that sits on the line
  (fake, as in N0-20b); sfk02 64.3 s ring hopping on the far touchline, can't tell (probably fake).
- added (2): sfk01 55.5 s light ball down the far touchline to a white player past the line = real; aik04 157.2 s ring
  wandering, can't tell.

## Seen, not fixed

AIK clip: dark has 0 entries while 32% of its ball time is in its final third - the ball dot crosses the line only
while the state says light has it (ball flicker). Same root cause as the other ball-limited stats: a steadier ball.

Tests: `tests/test_n0_24c_entries.py` (7: pass across loose frames, hold limit, no entry across the other team's ball,
flicker hop doesn't re-arm, hold 0 = the old loop on 30 random runs, app default, truth entries). Reaches the app at
the next run.
