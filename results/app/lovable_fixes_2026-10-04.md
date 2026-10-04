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
- What the app may show for a match is decided by `summary.ball_grade.possession_ok`, a field on the Supabase `matches` row:
  - `true` for SFK–BP and SFK–AIK
  - `false` for Vallentuna

**Expected numbers:**

| Match | Score (SFK–opponent) | Shots SFK / opp | Possession SFK | `possession_ok` |
|---|---|---|---|---|
| SFK – BP (1st half) | **1–0** | 3 / 4 | 46% | true |
| SFK – AIK (1st half) | **0–2** | 9 / 7 | 34% | true |
| SFK – Vallentuna (full) | **3–2** | 8 / 9 | hidden | false |

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

## 4. Hide what isn't verified (Vallentuna today)

When `summary.ball_grade.possession_ok` is **false**:
- **Hide** these tabs or cards:
  - Ball (layer and stats)
  - Pressing
  - Possession %
  - Losses / balls lost
  - Turnovers
  - Pass completion, passes, the pass map ("Where did our passes go?")
  - Field tilt
  - Set pieces
- **Show** a short note in their place: "Possession, passing and pressing are still being verified for this match."
- **Keep showing:**
  - Score
  - Shots and the shot map
  - The player layer on the video
  - Team heat maps
  - Shape (length, width, line height)

Rules for **every** match, including BP and AIK:
- Hide the **pass map and pass counts** ("535 located passes"). About 30% of the passes are not real yet. Pass completion % may stay for BP and AIK only.
- Hide **"Distance covered"** (50,541 m / 63,385 m). It only counts players while they are in the camera's view, so it is far below the real figure.
- Pressing for BP and AIK may stay, using the 2‑metre rule already agreed (`frames[].pressed`).

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
   - The pressing, ball, pass and possession tabs are hidden with the note.
   - Shot map: 8 SFK shots towards the right goal, 9 Vallentuna shots towards the left.
3. **BP and AIK:**
   - Possession and pressing are visible.
   - The pass map and distance covered are hidden.
   - The "2nd" half option is hidden.
4. **On a phone:** the tabs don't overlap, and the active tab and toggle are obvious at a glance.
