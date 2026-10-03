# Lovable prompt — match library for the demo (3 Oct)

Paste into Lovable as one message.

---

Hide every match from the library except the two demo matches. Do not delete anything.

- Show only these match ids, in this order:
  1. `SFKBP1109` — title "SFK – BP · first half"
  2. `p15u-vs-aik-2026-09-21-bd09` — title "SFK – AIK · first half"
- Every other row in the `matches` table is hidden from the library list, from search, from "recent", and from any dropdown or picker. Opening a hidden match by direct URL may still work; it just is not listed.
- Put the allow-list in one constant (`DEMO_MATCH_IDS`) at the top of the library data loader so it is one line to change. When the constant is empty, the library shows everything (today's behaviour).
- Keep the Beta label on both cards.
