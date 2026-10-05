# Lovable fixes for the Ipanema demo (Tue 6 Oct)

For the Claude chat that edits the Lovable app. Daniel looked at the app on his phone on Sunday evening and found the problems below. Fix them in the order listed, one Lovable message per section. After each one, open the three demo matches and check the "Expected" lines.

The data files are correct as of Sun 4 Oct, 19:30. Don't change any numbers in the app. Read them from the files as described here.

---

## Facts about the data (read before changing anything)

- Three demo matches (match ids):
  - `SFKBP1109`: SFK vs BP, **first half only**
  - `p15u-vs-aik-2026-09-21-bd09`: SFK vs AIK, **first half only**
  - `p15u-vs-vallentuna-2026-10-03-6cce`: SFK vs Vallentuna, **full match**
- In all three matches **team "A" = Sollentuna (SFK)** and **team "B" = the opponent**.
- Club names to show:
  - A = "SFK"
  - B = "BP", "AIK" or "Vallentuna"
  - Never show "P15", "P1", "VAL" or a "P15" placeholder logo.
- Goals and shots come from `stats.json → metrics.shots[]`. Each entry has:
  - `t`: seconds in the video
  - `team`: "A" or "B"
  - `x_m`, `y_m`: metres on a 106 × 64 pitch
  - `goal`: true / false
  - `outcome`: "goal" / "on target"
- Count only entries whose `t` is inside one of `match_data.periods[]` (`t_start - 5` to `t_end + 5`).
- Direction: in the data, **team A always attacks towards x = 106 (right)** and team B towards x = 0. The second half is already mirrored in the data, so never flip it again.

**Expected numbers:**

| Match | Score (SFK–opponent) | Shots SFK / opp | Possession SFK |
|---|---|---|---|
| SFK – BP (1st half) | **1–0** | 3 / 4 | 46% |
| SFK – AIK (1st half) | **0–2** | 9 / 7 | 34% |
| SFK – Vallentuna (full) | **3–2** | 8 / 9 | 33% (being re-checked) |

---

## 1. The score is wrong everywhere (shows 0–0)

- Library card, match header ("P15 0-0 VAL") and any other score display: compute the score from `metrics.shots[]` with `goal: true`, inside the periods, counted per team (A left, B right). Don't use a stored score field.
- Expected: BP 1–0, AIK 0–2, Vallentuna 3–2.

## 2. Team names and logos

- Show "SFK" with the SFK crest for team A in every match, in the header, toggles, legends, table headers, the card and the shot‑map legend ("SFK · only", "→ SFK attack").
- Show the opponent's real name for team B: "BP", "AIK", "Vallentuna" (short form "VAL" is fine only where space is tight).
- Remove the "P15" / "P1" placeholder boxes.

## 3. Add Vallentuna to the library

- Add `p15u-vs-vallentuna-2026-10-03-6cce` to `DEMO_MATCH_IDS` as the third match, titled "SFK – Vallentuna · full match", with the Beta label.

## 4. Show every stat (Daniel, 4 Oct 19:31: "don't hide stuff")

- Show all tabs and cards for all three matches: ball, pressing, possession, losses, turnovers, passes and the pass map, field tilt, set pieces, shape, distance.
- Keep the **Beta** label on the match.
- Don't add "not verified" notes and don't hide anything based on `ball_grade`. The data team is fixing the numbers at the source, and the app will pick up the new files automatically.
- Rename "Distance covered" to **"Distance covered (while in camera view)"**. The camera follows the ball, so players out of shot are not counted.

## 5. Shot map ("Where did shots come from?")

- With "SFK only" selected, show only `team == "A"` shots. With the opponent selected, only `team == "B"`. "Both" shows both teams, in two clearly different colours, with a legend.
- Plot `x_m, y_m` as stored. SFK shoots towards the right goal. Opponent shots should cluster at the left goal. Don't mirror anything.
- Goal = cream ring, on target = filled dot (as now).
- Count under the map = the shots of the selected team only. For Vallentuna: 8 (SFK), 9 (VAL), 17 (both).
- Expected: SFK shots cluster near the right goal, opponent shots near the left goal.

## 6. Buttons and tabs blend into each other (design)

On the stats screen, the tab row (BALL · PRESSING · SHAPE · SHOOTING · PLAYERS · SET PIECES · PASS…) and the toggles (SFK / Both / Opponent, Full / 1st / 2nd) are hard to tell apart:
- **Tab row:**
  - Equal padding (at least 12 px left and right) and 8 px gap between tabs.
  - Labels must never overlap ("SHOOTINGPLAYERSET PIECESPASS" currently runs together).
  - Make the row scroll sideways on phones, with a fade at the right edge so it's clear there are more tabs.
- **Active state:**
  - Selected tab or segment gets a filled background in the accent colour with dark text.
  - Unselected ones are transparent with muted text and a 1 px border.
  - Don't use an outline‑only active state; it blends in on the dark background.
- **Segmented toggles:** clear 1 px dividers between the options, and contrast of at least 4.5:1 between the active and inactive text.
- Touch targets at least 44 px high.
- **Bottom navigation** (Insights / Match / Stats / Phases): the active item gets the accent colour plus the line above it. The others stay muted.
- Check at 390 px wide (iPhone) that nothing overlaps.

## 7. First half / second half

- BP and AIK are first halves only. Hide or disable the "2nd" and "Full" options there, or show "First half only".
- For Vallentuna, "1st" and "2nd" split by `match_data.periods[]`:
  - First half: 435–2890 s
  - Second half: 3220–5645 s
  - Don't split at duration / 2.

---

## Checks after the changes (do all of them)

1. **Library:** three cards showing 1–0, 0–2 and 3–2, team names "SFK" and the opponent, all "Ready" with Beta.
2. **Vallentuna:**
   - Header reads "SFK 3–2 Vallentuna".
   - All tabs are visible.
   - Shot map: 8 SFK shots towards the right goal, 9 Vallentuna shots towards the left.
3. **BP and AIK:**
   - All tabs are visible.
   - The "2nd" half option is hidden.
4. **On a phone:** the tabs don't overlap, and the active tab and toggle are obvious at a glance.


---

## 8. Bugs Daniel found on Sunday night (fix these first)

1. **Score in the match header must be the score at the current video time,** not the final score.
   - Count `metrics.shots[]` with `goal: true` and `t` up to the current playhead, per team.
   - Vallentuna: 0–0 until 60:04, then 1–0, 1–1 (62:22), 1–2 (64:00), 2–2 (91:53), 3–2 (93:13).
2. **The 2D view is empty** although the frames have positions.
   - Draw every `frames[].players[]` entry at `m = [x, y]` (metres on a 106 × 64 pitch, x to the right), coloured by `team`. Draw the ball at `ball.m`.
   - Second-half frames are already mirrored: don't flip them.
   - Example: Vallentuna at 54:47 (t = 3287 s) has 5 players; the 2D view must show 5 dots.
3. **Pins sit below the players,** more so near the bottom of the picture. The pin position is `players[].px`, in pixels of a **1920 × 1080** frame.
   - Scale it with the *displayed* video rectangle: `x * shownWidth / 1920`, `y * shownHeight / 1080`, offset by the letterbox or crop.
   - If the video element uses `object-fit: cover`, either switch it to `contain` or apply the same crop to the overlay.
   - Check: a pin should sit exactly at a player's feet in all four corners of the picture.
4. **Shots without a position.** From now on shots have `x_m`/`y_m` = `null` until their origin is marked by eye. On the shot map, don't draw shots with a null position. Under the map write "8 shots · positions being added" (the count still includes them). Never place a null-position shot on the goal line.
5. **"GK" labels:** show "GK" only where `players[].gk` is true. After tonight's data fix that is at most one player per team.

---

## 9. Monday 01:50: what Daniel still sees

1. **The shot map reads an old file.** The app says "17 of 17 shots off the pitch, x_m −3 to 109". The files on the server (since Sun 23:30) have `x_m`/`y_m` = `null` for every shot.
   - Bypass the cache: re-download `stats.json` and `match_data.json` when the match is opened (add `?v=<matches.updated_at>` to the file URL).
   - Shots with null positions: list them in a simple **shot timeline** (minute, team, goal / on target), so there's still a visual. Show the pitch map only for shots that have `x_m`/`y_m`. Positions are being added by eye.
2. **Don't hide anything.** Remove the notes "Pressing / Ball / Passes is still being verified for this match …". Show those tabs with their numbers for every match, with the Beta label (Daniel's decision, section 4).
3. **The debrief quote "Press faster when we lose the ball."** appears on every match. Our data files never contain that text, so it comes from the app.
   - Either build the debrief from this match's numbers, e.g. "We won the ball back within 5 s in X% of losses (opponent Y%)" using `teams[].regained_within_5s_pct`,
   - or remove the quote.
   - Never show the same fixed sentence on every match.

---

## 10. Everything over time (Daniel, 5 Oct 23:02)

Possession, pressing, shape and the other stats must be shown **over time** as well as totals. The data is already in `stats.json`, so no data change is needed. Add a "Match timeline" card at the top of the Stats screen, plus a small timeline inside each tab.

- **Shared x-axis:** match time in minutes. Only draw the parts inside `match_data.periods[]`, with a gap at half time. Show goal markers (`metrics.goals[]`, a small ball icon at `t`, in the team's colour) on every timeline. Add a vertical playhead line at the current video time; tapping a point on the timeline jumps the video there.
- **Possession over time:**
  - Source: `metrics.tilt_windows[]`, which has `t`, `t_end` and `possession_A` (0–1, the SFK share in that 15 s window).
  - Smooth over 5-minute rolling windows and draw an area chart around 50%: SFK above, opponent below, in the two team colours.
  - Skip windows with `possession_A` null and windows outside the periods.
  - Also show 5-minute bars: "SFK 58% · 0–5′".
- **Field tilt over time:** same windows, `tilt_A`. Skip null values.
- **Pressing over time:**
  - Count `metrics.pressure_points[]` per 5 minutes for each `pressing_team`, as two lines.
  - Add `metrics.high_turnovers[]` per 5 minutes per `team` as dots.
- **Shape over time:** from `metrics.shape_timeline.A` and `.B` (`t`, `length`, `width`, `line_height`), take the 1-minute medians and draw three small line charts (team length, team width, defensive line height), SFK vs opponent.
- **Passes over time:** count `passes[]` per 5 minutes per `team`, with the completed share (`completed`) as a thin line.
- **Sequences:** a strip of coloured blocks from `sequences[]` (`t_start` to `t_end`, colour by `team`), showing who had the ball and for how long.
- **Halves:** the Full / 1st / 2nd toggle filters every timeline.
- **Design:** dark theme as now, team colours consistent everywhere, readable at 390 px wide, axis labels in minutes (0′, 15′, 30′ …).
- **Check:** for Vallentuna, possession over time must show data in both halves (07:15–48:10 and 53:40–94:05), and the five goal markers must appear at 60:04, 62:22, 64:00, 91:53 and 93:13.
