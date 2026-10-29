# N0-29c (29 Oct, worker, local only, $0, nothing on Modal)

**Nightly card:** 29 Oct card, worse than 28 Oct = nothing. **Queue:** M2 (main session) and PC2 (waits for M2) skipped.

## What was checked
Per-player **top speed** and **high-intensity distance** (`stats.json players[].top_speed_ms`, `high_intensity_m`,
hidden in the app). N0-22c fixed distance but left both "still too high".

Truth: Metrica pro tracking, games 1 and 2, first 20 min, 22 players each. Top speed = best 1-s average (the usual GPS
definition); high-intensity = distance above 5.5 m/s. Each player's track was cut into seen runs / gaps like our app
clips (median seen run 0.6 s), our feet jitter added, and id swaps after 25% of gaps (our rate) or none (= what M2 gives).
`tools/n0_29c_speed.py` -> `metrica_speed.json`.

## Top speed: fixed
The old figure was the fastest single row: **9.9 m/s on every Metrica player** (the 10 m/s cap), so it said nothing
(rank vs the truth -0.27 to 0.19). New: the best 1-s average over time the player was seen the whole time (one seen run;
a raw jump > 3 m in <= 3 frames = another person, splits it).

| | truth median | old: median / error / rank | **new: median / error / rank** |
|---|---|---|---|
| game 1, no swaps | 8.12 | 9.9 / 1.8 / -0.24..-0.13 | **7.7-7.8 / 0.08-0.10 / 0.55-0.84** |
| game 2, no swaps | 7.20 | 9.9 / 2.7 / 0.09-0.14 | **6.9-7.1 / 0.29-0.30 / 0.55-0.70** |
| game 1, our id swaps | 8.12 | 9.9 / 1.8 / -0.27..-0.09 | **7.8 / 0.63-0.67 / -0.14..0.15** |
| game 2, our id swaps | 7.20 | 9.9 / 2.6 / -0.17..0.19 | **7.1 / 0.71-0.82 / -0.06..0.04** |

(error = median per-player m/s off.) Also tried: 95th percentile (4 m/s, far too low), 0.5 s / 2 s windows, no jump
split (0.5 s and no-split about as good; 1 s kept because it is the truth's own definition and the split guards real
mid-run id slides the simulation doesn't make).

Real app clips (`tools/n0_29c_clips.py` -> `app_clips.json`): tracks over 8 m/s 55% / 49% -> **0%**; median of tracks
seen >= 20 s 9.5 -> **5.2 m/s** (SFK-BP), 9.3 -> **4.3** (AIK). By eye on the 8 longest tracks (`speed_*.png`, red =
old row speed, blue = new): the old top was always set by a spike cluster (a jump / box flicker, 8-10 m/s), the new one
sits on a real, held run (5.2-6.4 m/s); one AIK track reads 3.9 because its only fast bit is such a spike cluster.

So: numbers are now believable, but **who is fastest needs one track per player (M2)** - with our id swaps the order
is no better than chance. Stays hidden in the app. `IPANEMA_TOP_SPEED=old` = old. Reaches the data at the next run.

## High-intensity distance: no change
Tried counting only stretches held above 5.5 m/s for >= 1 s / 0.5 s, or seen steps only: **15-50% of the truth**
(our seen runs are too short to hold 1 s) vs the old 103-150%, and worse per player (51-89% vs 13-55% off). Old kept
(`IPANEMA_HI=old` is the default; `high_intensity()` is there for the lab). Stays hidden.

## Tests
`tests/test_n0_29c_top_speed.py` (9): steady run reads its speed, still player with jitter / a 2.5 m box slide / a 4 m
jump don't make a sprint (old does), a burst under 1 s or split by a gap doesn't count, stats uses new + old switch,
high-intensity stays old, lab dry run. Suite: 762 pass (+9); 3 fail only because this machine lacks `supervision`
(same before the change).
