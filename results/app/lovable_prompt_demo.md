# Lovable — demo changes (3 Oct). Paste as ONE message.

---

Hide every match from the library except the two demo matches. Do not delete anything.

- Show only these match ids, in this order:
  1. `SFKBP1109` — title "SFK – BP · first half"
  2. `p15u-vs-aik-2026-09-21-bd09` — title "SFK – AIK · first half"
- Every other row in the `matches` table is hidden from the library list, from search, from "recent", and from any dropdown or picker. Opening a hidden match by direct URL may still work; it just is not listed.
- Put the allow-list in one constant (`DEMO_MATCH_IDS`) at the top of the library data loader so it is one line to change. When the constant is empty, the library shows everything (today's behaviour).
- Keep the Beta label on both cards.


Two changes, both about showing only what is right.

**1. Match section / event list: show only verified events.**

- List only events with `type` `goal` or `shot`. Do not list `set_piece`, `turnover_lost`, `turnover_won`, `sequence_end`, `better_option`, `high_turnover`, `pass_risky`, `pass_bad`.
- Newer data files carry `tier` on every event (`verified` / `beta` / `hidden`) and `verified: true|false`. When `tier` is present, list `verified` by default; put `beta` behind a small "Show beta events" toggle (off by default); never list `hidden`. When `tier` is absent (older files), use the type rule above.
- The counts shown next to the list (e.g. "12 set pieces") must follow the same rule: count only what is listed.

**2. Pressing screen: use the pipeline's 2-metre rule, not 5.**

- A frame counts as "pressure on the ball carrier" when `frames[].pressed` is `true`. If `pressed` is absent (older files), use `pressure_m <= 2`. Never use 5 m for "pressed".
- `pressure_m` is metres from the ball carrier to the nearest opposing player, `null` when nobody is on the ball.
- Which side is pressing: the carrier's team is the team of the player whose `id` equals `carrier` in that frame's `players[]`; the presser is the other side. Do not use `possession` for this.
- "Where we pressed them" = frames where the carrier is the opponent and `pressed` is true; "where they pressed us" = frames where the carrier is ours and `pressed` is true. Plot at the carrier position (`players[]` entry with `id == carrier`, its `m`).
- Second half: `periods[]` has `mirrored: true` for the second period; positions in those frames are already mirrored so both halves attack the same way — aggregate without flipping again.
- The headline "pressed within 2 s" uses the same 2 m rule (`turnover_lost.payload.time_to_press <= 2`), so the map and the headline now agree.
- Per-event values for the "moments that failed" lists, present in newer files: `turnover_lost.payload.near_at_2s` (count), `pressed_within_2s` (bool); `turnover_won.payload.forward_within_3s` (bool), `lost_back_5s` (bool). When absent (older files), show the aggregate only.

Keep the Beta label on the match.

**3. Events outside the video.**

- Ignore any event (and any shot in `stats.metrics.shots`) whose `t` is greater than the video's duration, or that falls outside every `periods[]` window (`t_start - 5` to `t_end + 5`). The first-half files currently carry second-half goals and shots from Veo's list; they must not appear in the list, the goal count or the shot map.
