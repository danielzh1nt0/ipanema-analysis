# B4: bigger ball answer key without clicking (1 Oct 2026)

**Result: 287 new checked ball moments on the SFK-BP clip (34 -> 321 with the old key). Free, no Modal.**

How:
1. `tools/b4_autokey.py` (local): frames where the app picker (RF-DETR + WASB 30 peaks, v2), RF-DETR's top guess (score >= 0.4)
   and WASB's top guess (>= 0.5) are all within 10 px. At least 0.5 s apart, not next to an old key moment -> 312 candidates.
   On the 34 old moments this rule fires 20 times and is right 20/20.
2. `tools/b4_sheet.py` (free runner): picture sheets, close-up + wide view per moment -> `results/free/b4/sheet_00..12.jpg`,
   plain 64x64 crops in `results/free/b4/crops.npz` (usable as ball pictures for training).
3. Claude checked all 312 by eye -> `key.json`:

| Verdict | Count |
|---|---|
| ball (guess is on the ball) | 287 |
| can't tell (in a crowd / at feet, hidden) | 9 |
| a spare ball lying at the goal post, not the match ball | 16 |

Found on the way:
- **The picker follows a spare ball lying at the left goal post** for ~4 s at a time, three times in 5 min
  (145-149 s, 174-175 s, 250-254 s; sheet 6 #161-167, sheet 7 #180-181, sheet 10 #253-259). In #260 both balls are in view.
  The static-clutter rule does not drop it because players walk past it.
- 13 moments (#240-252) are the match ball lying near the touchline before a restart; kept as "ball".

Limits (read before using the numbers):
- The key leans easy: every moment is one where both finders already agreed. It is good for catching a picker that gets
  worse and as training pictures, not for measuring the hard cases (in the air, at feet, crowds; see results/picker/misses.md).
  Keep reporting the 34-moment score for that.
- One clip, one ground. Moments are 0.5 s apart, so neighbours are not independent.
