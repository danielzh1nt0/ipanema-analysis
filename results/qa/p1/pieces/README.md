# P1 striped kits on the real tracking pieces (1 Oct)

**Why players were still missing:** the P1 v2 switch (far pair + mean shirt colour) only fired when the default kit fit
left out a colour group at >= 1.0 x the team gap. On the 21 real 60-s pieces (7 grounds x 3) that number is 0.40-0.90,
so it never fired, and near striped Spånga players stayed "neither team" (99 of 633 people on the pitch in piece 1125,
almost all of them clear, near Spånga players).

**Fix (v3, now the default):**
1. New switch rule: use far pair + mean colour when the far-pair rule picks a *different pair of teams* than the default
   (a team colour moves by more than 18). Old rule kept as a second reason. Off: `IPANEMA_KIT_AUTO=0`.
2. The far-pair rule now picks the second team with lightness counting half (like its merge step). Before, a big group
   of dark people (spectators in coats, shade) could beat the real second kit. Old rule: `IPANEMA_FAR_PICK=plain`.

**Where it switches:** all 3 Spånga pieces and Djursholm 2714 (there the default made spectators a team and read the
red team as "neither"). Nowhere else.

**Graded by eye** on the sheets in this folder (one per piece: only the people whose reading changes when far+mean is
used, rows "neither -> team", "team -> neither", "A -> B", "B -> A"):
- Spånga: ~260 people, nearly all near striped players, go from "neither" to their team; 44 striped players in piece 4050 go the other way;
  "neither" goes 99 -> 8, 54 -> 1, 113 -> 44. Cost: referees and keepers mostly read as a team on Spånga now (they were
  mostly in a team before as well).
- Djursholm 2714: 220 red players get their team, 403 spectators leave team A. Clearly right.
- Pieces where far+mean was mixed and the switch stays off: Solberga 2766 (referees in light blue would join a team),
  Solberga 1456 (12 white players would become "neither"), Solheim 2431 (spectators would join), SFK (players gained
  but also grey-shirt people at the touchline). Reymersholm, Vasalund, Solberga 5242 looked better with far+mean but
  are left alone for now (possible next step).

**By-eye keys (kitprobe frames, p1lab_v3.json):** Spånga 181 -> 233/279 right, Reymersholm 239/252 unchanged, SFK unchanged.

**Tracking check (free runner, started 1 Oct):** 20 s tracked with the switch off/on on Spånga (minute of piece 1125)
and Djursholm (piece 2714) -> results/qa/p1_track_{spanga,djursholm}_{off,on}/.

Re-run: `python tools/p1piecelab.py` (local, ~1 min). Tests: tests/test_p1_gate.py.

| piece | before A/B/neither | now A/B/neither | switched |
|---|---|---|---|
| SFKBP1109_s1200_120 | 299/224/225 | 299/224/225 | - |
| SFKBP1109_s1200_20 | 297/249/296 | 297/249/296 | - |
| SFKBP1109_s1200_220 | 272/229/200 | 272/229/200 | - |
| p09-norrviken-vs-solheim-2026-08-30_1279 | 292/275/168 | 292/275/168 | - |
| p09-norrviken-vs-solheim-2026-08-30_2431 | 309/255/178 | 309/255/178 | - |
| p09-norrviken-vs-solheim-2026-08-30_4607 | 243/271/83 | 243/271/83 | - |
| p15u-vs-djursholm-2026-09-26_1428 | 294/334/337 | 294/334/337 | - |
| p15u-vs-djursholm-2026-09-26_2714 | 348/301/299 | 217/249/482 | yes |
| p15u-vs-djursholm-2026-09-26_5143 | 303/285/365 | 303/285/365 | - |
| p15u-vs-reymersholm-2026-09-18_1190 | 210/165/90 | 210/165/90 | - |
| p15u-vs-reymersholm-2026-09-18_2262 | 310/207/55 | 310/207/55 | - |
| p15u-vs-reymersholm-2026-09-18_4286 | 219/155/83 | 219/155/83 | - |
| p15u-vs-spanga-2026-09-25_1125 | 307/227/99 | 326/299/8 | yes |
| p15u-vs-spanga-2026-09-25_2137 | 224/181/54 | 196/262/1 | yes |
| p15u-vs-spanga-2026-09-25_4050 | 303/203/113 | 335/240/44 | yes |
| p15u-vs-vasalund-2026-09-20_1130 | 226/225/65 | 226/225/65 | - |
| p15u-vs-vasalund-2026-09-20_2147 | 181/138/98 | 181/138/98 | - |
| p15u-vs-vasalund-2026-09-20_4068 | 231/196/77 | 231/196/77 | - |
| solberga-vs-p09-norrviken-2026-09-11_1456 | 243/263/39 | 243/263/39 | - |
| solberga-vs-p09-norrviken-2026-09-11_2766 | 204/243/164 | 204/243/164 | - |
| solberga-vs-p09-norrviken-2026-09-11_5242 | 150/232/193 | 150/232/193 | - |
