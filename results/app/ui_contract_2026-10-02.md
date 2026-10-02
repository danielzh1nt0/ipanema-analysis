# UI contract ↔ pipeline export (2 Oct)

Checked against the real exports of AIK (p15u-vs-aik-2026-09-21-bd09) and SFK-BP (SFKBP1109), contract "1.1",
written by ipanema/export.py. "OK" = the key exists under that name with that meaning. Demo verdicts are from the
stats audit (claude/stats-audit-2026-10-02.md): SHOW / Beta / HIDE.

## The four that carry the most weight (§1.8)

| UI needs | What the export has | Action |
|---|---|---|
| `time_to_press` on `turnover_lost` | OK, `payload.time_to_press` (s) — set on 156 of 316 losses on AIK, `null` when no opponent came within 5 m | None. The median (1.17 s) is far too fast vs real football (3-5 s): it comes from picker flicker, so the Pressing headline is **HIDE** for the demo. |
| `players[].state` on frames | Always `"observed"`. The pipeline never writes a guessed player position: a player is in a frame only when the detector saw him there. There is nothing to mark `stale`. | None. Treat absence as "not seen", not as stale. |
| `block_length_median_m` | OK, in `teams[]` (AIK A 25.2 m) | Shape values only use the players in view (~30% of the pitch), so they are "shape of the visible group". Beta label or HIDE. |
| `summary.ball_grade` | **Not in `stats.teams[]`.** It is in the Supabase `matches` row, column `summary` → `summary.ball_grade.{possession_ok, events_ok, accuracy, near_player_pct}`, and in `matches/<id>/summary.json`. Also `summary.ball_reliable`. | UI must read it from the matches row (or we add it to `teams[]` at the next run — one line, but needs a run). |

## 1.2 Team row — `stats.teams[]`

| UI key | Export | Demo |
|---|---|---|
| possession_pct | OK | SHOW (Beta) |
| possession_s | OK | SHOW |
| sequences | OK (1,867 for A on AIK — flicker) | HIDE |
| passes | OK (946) — ~30% are fake, see B10 | HIDE count |
| pass_completion_pct | OK (71) | SHOW (Beta) |
| passes_per_sequence | OK (0.5 — meaningless while sequences flicker) | HIDE |
| pressed_within_2s_pct | OK (30) | HIDE |
| regained_within_5s_pct | OK (82) | HIDE |
| time_to_press_median_s | OK (1.17) | HIDE |
| near_at_2s_median | OK | HIDE |
| pressures_applied | OK (855 — ~5x real) | HIDE |
| ppda_opp_passes_per_def_action | OK (1.2 — real is 7-15) | HIDE |
| block_length_median_m, block_width_median_m, def_line_height_median_m | OK | Beta ("visible players") |
| distance_m_total_visible | OK (63,079 m) | Beta ("visible players") |
| forward_pass_share_pct | OK (44) | Beta |
| progressive_passes | OK (116) | Beta |
| better_option_count | **not in teams[]** — only per player (`players[].better_option_count`) and as `better_option` events (675 on AIK, unverified) | HIDE anyway |
| summary.ball_grade / ball_reliable | not here — see above | — |
| duration_s | not in teams[]; it is `library.json` / matches row `duration_s`, and `match_data.periods[]` has `t_start`/`t_end` per half (**use periods for the 1st/2nd-half split**, not duration/2: AIK 1st half is 366-3390 s, 2nd 3950-6899 s) | — |
| attack_right | not in teams[]; `match_data.attack_right` {A,B} and per period in `match_data.periods[].attack_right` (2nd half is `mirrored: true`) | — |

Extra keys we write that the UI does not read: `losses`, `recoveries`, `time_to_forward_pass_median_s`, `forward_within_3s_pct`, `lost_back_5s_pct`.

## 1.3 Player row — `stats.players[]`

**`id` is a track id, not a shirt number** (e.g. 100474 on AIK, 1 on SFK-BP). A track is a 2-130 s piece of one
person; the same player appears as 50+ rows. Nothing in the export knows shirt numbers. The whole player table is
HIDE until one-ID-per-player (M2, paused).

| UI key | Export |
|---|---|
| id, team, touches, passes, passes_completed, better_option_count, distance_m, time_visible_s | OK |
| risky_passes / passes_risky | **neither** — we write `pass_quality: {good, risky_completed, bad_lost, execution_error}`; risky = `pass_quality.risky_completed` |

Extras not read: avg_x_m, avg_y_m, passes_forward/backward/sideways, passes_received, progressive_passes, pct_att/mid/def_third, time_on_ball_s, time_under_pressure_s, pressures_applied, counterpress_rate, losses_seen, high_intensity_m, top_speed_ms (**top_speed_ms is not trustworthy, do not show**).

## 1.4 Metric buckets — `stats.metrics{}`

| UI key | Export |
|---|---|
| shots[] {team, x, y, goal, on_target} | `shots[]` has `team, x_m, y_m, goal, outcome ("on target"/"goal"), distance_m, t, source`. **Keys are `x_m`/`y_m`, not `x`/`y`; `on_target` is `outcome in ("on target", "goal")`.** All shots come from Veo's highlight list (`source: "veo"`); our own detected shots are in `shots_detected[]` (6 on AIK, not for the demo). |
| field.{A,B}.field_tilt_pct | OK (A 24 on AIK) |
| field.{A,B}.entries_count | **not there** — we write `field.{A,B}.final_third_entries[]` (list of {t, how, y_m}); count = its length |
| high_turnover_counts.{A,B} | OK (32/29) — HIDE (too high) |
| shape_timeline.{A,B}[] {t, length, width} | OK, 1 row per second, plus `line_height`, `n`; values are `null` when fewer than ~4 players seen |

Extras: `goals[]`, `pass_network.{A,B}.edges[]` (track ids — meaningless until M2), `runs[]`, `pressure_points[]`, `tilt_windows[]` (15-s windows, `tilt_A`, `possession_A`), `veo_rejected[]`, `contract`.
Also top-level `stats.heatmaps{track_id: 12x8 grid}` (per track, so per-player heat maps are HIDE) and `stats.restarts[]` (HIDE), `stats.grid = [12, 8]`.

**Team heat map:** not exported as such. Build it from frames: all `players[].m` of a team per `stats.grid`, or sum `heatmaps` over the team's track ids (team is in `players[]`). Team-level is fine to SHOW (Beta).

## 1.5 Pass records — `stats.passes[]` — the one spelling we write

`{t, from, to, team, completed (bool), from_m [x,y], to_m [x,y], gain_m, length_m, kind (forward/backward/sideways), progressive, lane_open, quality, missed_open_forward, better_option}`

- Time: **`t`**. Passer/receiver: **`from` / `to`** (track ids). Points: **`from_m` / `to_m`**.
- Completed: **read `completed`**. Do not derive it from `quality`: `quality` is `unknown` for 911 of 1,943 AIK passes (incl. most incomplete ones), so "quality not in {bad_lost, incomplete}" would count them as complete. `bad_lost` only occurs 2 times.
- Quality values that occur: `good`, `risky_completed`, `bad_lost`, `execution_error`, `unknown`.

Drop: `time`, `start_t`, `player_id`, `receiver_id`, `start`, `end`, `outcome`.

## 1.6 Events — `match_data.events[]`

Shape OK: `{id, t, type, team, title, subtitle, payload}`. Types we write (AIK counts): `sequence_end` 1088, `better_option` 675, `turnover_lost` 316, `turnover_won` 316, `set_piece` 88, `high_turnover` 61, `shot` 23, `pass_risky` 16, `goal` 7, `pass_bad` (0 here, exists). **Never written:** `shot_blocked`, `interception`, `pass_intercepted`.

| Event | Payload we write |
|---|---|
| any | **no shirt key** (shirt / shirt_number / player_shirt are never written). Positions: `x_m`/`y_m` on shots & set pieces; `from_m`/`to_m` on passes; `ball_m_before_press` on losses. No `x/y`, `px/py`, `start_x`. |
| turnover_lost | `time_to_press` (s or null), `near_at_2s`, `regained_within_5s` (bool), `ball_m_before_press`, `reactions {pressed[], jogged[], stood[]}` |
| set_piece | `kind` ∈ `throw-in`, `free kick`, `goal kick`, `corner` (note the space and hyphen), `team`, `x_m`, `y_m`, `t` |
| better_option | `from`, `to`, `best_to`, `from_m`, `to_m`, `best_gain_m`, `best_bypassed`, `best_space_m`, `best_open`, `played_value`, `best_value`. **No** `better_x/y`, `target_x/y`, `played_x/y`: the better receiver's position is not in the payload (look him up in the frame at `t`). |
| shot / goal | `outcome` ("on target" / "goal"), `goal` (bool), `x_m`, `y_m`, `distance_m`, `source`. **No `on_target` bool** — derive from `outcome`. |

## 1.7 Frames — `frames_NNN.json` chunks (300 s each, listed in `match_data.frame_chunks[]`; `match_data.frames` is empty for full matches)

`{t, players[], ball, possession, phase, carrier, pressure_m, near_opps, shape, lanes, pitch_lines, cal_ok}`

- `players[]`: `{id, team, gk, state: "observed", conf, px, m}` (+ `kmh`, `dist_m` when the speed layer is on — it is off). Always observed; see §1.8.
- `ball`: `{px, m, state, conf}` or `null`. **`state` is `observed` or `bridged`** — not predicted/stale. `bridged` = we did not see the ball and filled the gap between two sightings. Treat `bridged` as your `predicted` (draw faint or skip).
- `shape.{A,B}`: `{hull_m, n, length, width, line_height}` (null when too few in view).
- `pitch_lines`: the 3x3 calibration matrix flattened (9 numbers, pitch metres → pixels), or `null` when the frame's calibration is unsure; `cal_ok` says the same as a bool.
- `possession`: "A"/"B"/null; `phase`: "control" or the loose-ball state name.

## Part 2 — tier 1, what already exists

| # | Item | In the export today? |
|---|---|---|
| 1 | before/after split of the headline figures | Not pre-computed. Events and passes carry `t`; `tilt_windows[]` already gives possession per 15 s. Client-side is possible for possession; press/shape figures need the turnovers/shape_timeline (both have `t`). Doable client-side; or we add it at the next run. |
| 2 | time_to_press distribution | **Already there**: every `turnover_lost.payload.time_to_press`. No change needed. |
| 3 | turnover map by zone | Not pre-computed; `ball_m_before_press` on `turnover_lost` gives the position (null for some). `match_data.turnovers[]` has the same with `t`. |
| 4 | turnover → shot within 15 s | Not computed; shots (`metrics.shots[].t`, from Veo) and turnovers both have `t`. Client-side is easy. |
| 5 | set-piece outcome | Not computed; `set_piece.t` + frames `possession` + shots give it. But restarts are HIDE (168 vs Veo 80), so wait. |
| 6 | rest defence | Not computed; frames have `players[].m`, `ball.m`, and `attack_right`. Only players in view (~30% of pitch) → the count would be "visible goal-side players", mostly meaningless. Do not build until the camera coverage problem is solved (it is not solvable with one follow-cam). |
| 7 | build-up exit rate | `sequences[]` has `t_start/t_end`, `gain_m`, `reached_final_third` but **not the start zone**; `start`/`end` are frame numbers, so the start position is in the frame. Blocked by sequence flicker (HIDE). |
| 8 | shot distance / inside-box share | **Partly there**: `metrics.shots[].distance_m` (and `x_m,y_m`). Inside-box = client-side from x_m/y_m. Shots are Veo-located, so fine to show. |
| 9 | possession ladder | Not computed; blocked by the sequence flicker. |

Tier 3 #16 (aerial duels): out of reach — no ball height from one camera. #18 (sub minutes): out of reach until M2.

## Open with the UI

1. Read `completed` on passes, `x_m/y_m` on shots, `outcome` for on-target, `periods[]` for halves, `ball_grade` from the matches row.
2. Treat `bridged` ball state as predicted; treat player absence as not seen (no stale state exists).
3. `players[].id` is a track id: hide anything per player and the shirt-number idea until M2.
4. Shirt numbers are never written; the "any" payload shirt keys will always be empty.
