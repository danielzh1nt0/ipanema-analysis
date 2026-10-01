# Lovable prompt: show all stats with a Beta label (1 Oct)

Paste this into Lovable:

---
Right now the match page hides some stats and layers when the match data says they are not reliable enough. Change that for the demo:

1. Always show every stat, chart and video layer (ball, possession, passes, turnovers, sequences, shots, events list, per-player stats), even when the match summary says it is withheld. The flags that currently hide things are in the match's `summary`: `ball_reliable`, `ball_grade.possession_ok`, `ball_grade.events_ok`, and `summary.quality.<stat>.ok`. Stop hiding on these flags.
2. Put a small "Beta" badge next to the title of every stats section and on the layer toggles. Style: subtle pill, muted colour, same on light and dark.
3. Hover/tap on the Beta badge shows: "Early version - computed automatically from the video, may contain errors."
4. Where the flag above is false, use a slightly stronger badge text "Beta - low confidence" instead of "Beta".
5. Don't change any numbers, only what is shown and the labels.
---
