# Lovable — library cards (3 Oct evening)

Paste as one message.

---

Three fixes on the match library cards, nothing else:

1. Remove the "balls lost" figure from the card subtitle. Show possession and shots only, e.g. "46% possession · 4 shots".
2. The scoreline on the card must come from the data, not a stored score: count events of type `goal` in `match_data.events` whose `t` lies inside one of `match_data.periods[]` (`t_start - 5` to `t_end + 5`), per team. Expected: SFK – BP first half reads 0–1, SFK – AIK first half reads 0–2. Shots on the card: same rule, type `shot` or `goal`, inside the periods (SFK–BP 7, AIK 16 — both teams together), or show the home team's shots only if that is the design; say which in the label.
3. Status label: show "Ready" (or nothing) when the match has `stats` and `match_data` files; "Setup needed" only when they are missing. Both demo matches are ready.
