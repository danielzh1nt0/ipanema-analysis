# Lovable prompt — scores on the cards + Vallentuna in the library (4 Oct)

Paste into Lovable as one message.

---

Two fixes on the match library, nothing else.

**1. Add Vallentuna to the library.**
Add `p15u-vs-vallentuna-2026-10-03-6cce` to `DEMO_MATCH_IDS` as the third match, title "SFK – Vallentuna · full match". Keep the Beta label. Keep the other two as they are.

**2. The scoreline on every card must come from the match's data, not a stored score.**
- Read `stats.json` → `metrics.shots[]`. A goal is an entry with `goal: true`. Count goals per `team` ("A" / "B").
- Only count entries whose `t` lies inside one of `match_data.periods[]` (`t_start - 5` to `t_end + 5`).
- In all three matches **Sollentuna (SFK) is team "A"** (the darker kit) and the opponent is team "B". Left side of the card = SFK = team A goals; right side = opponent = team B goals.
- Shots on the card: entries in `metrics.shots[]` inside the periods, both teams together.
- Expected after the data refresh tonight: **SFK – BP first half 1–0**, **SFK – AIK first half 0–2**, **SFK – Vallentuna 3–2**.
- Status "Setup needed" only when `stats.json` or `match_data.json` is missing; otherwise "Ready".
