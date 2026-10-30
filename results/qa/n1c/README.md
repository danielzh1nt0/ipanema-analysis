# N1c (30 Oct, worker, local + free runner, $0, nothing on Modal): AIK team labels - keepers, referee, field players

**Why:** N0-30b found AIK's possession reads too high for SFK (export 45.5%, by eye 32%, 23-43%) because AIK players carry SFK's
label: the red keeper (3/3 times he was the carrier), some white players, and the referee usually gets a team.

## How
- `tools/n1c_kitread.py`: reads every exported player's shirt colour (dark SFK / white AIK / red / light-blue referee / can't tell)
  from a torso box above his foot point. Free runner read the whole match 3 times (`tools/n1c_kitvote_run.py`, 6,446 rows every 0.9 s,
  ~95,000 reads, 8 min each): `results/free/n1c` (box sized from the foot's row), `n1c2` (box sized from the row's ground scale -
  Veo zooms, the row alone was 0.65-1.0x too short and put the referee's box on his black shorts), `n1c3` (+ a shirt-vs-grass
  lightness check: far black shirts in hazy sun read blue-grey like a white shirt in the shade).
- Repo frames (150 full-size): the referee now reads blue 35/35 tiles (was 10, the rest read 'dark'); `sheet_*.jpg`.
- `tools/n1c_relabel.py`: a player's label from the reads of his id within +-5 s (export ids slide from one player to another over a
  long life, so a whole-match vote moved both - e.g. the keeper's id ending on a dark player). Possession is then edited like for
  like: the export's possession moves with the new label of the player who made it (carrier, else the nearest player of that team);
  a re-computation of possession_simple from frames_lite copies only 60-70% of the export rows, so it is reported but not used.
  Scored on the 75 decided by-eye moments of N0-30a + N0-30b (54 where the export has a team at the nearest row).

## Finding: the keepers' shirts do not say their team on this match
By eye, the red keeper stands in the picture's RIGHT goal in the first half with darks around (p25 goal kick; N0-26b's kick-off picture
puts SFK in the right half) - SFK's goal - and in the second half there is a red keeper in the right goal (AIK's: q03, q09, q17, q65,
q76) AND one in the left goal (SFK's: q01, q02, q21-q28 at 4038-4778 s; orange there at q30). So **a keeper takes the team of the goal
he stands in**: first half right goal = SFK, left = AIK; second half the other way (export x runs from the picture's right goal, x~0,
to the left one). The export's own keeper flags follow `attack_right_A=true` (mirrored, the open N0-26b question) - 2nd half: the
right-goal keeper is labelled SFK 163 rows, the left one AIK 140. With the goal rule every by-eye keeper moment where the export has the keeper is right
(q09, q17, q65, q76, p25; the shirt rule 'red = AIK' gets p25 wrong); at q03 the keeper is not in the export row at all.

## Result (by-eye moments right; export's whole-match SFK share; by eye 32%, 23-43%)
| rule (reads n1c3, +-5 s) | right at the nearest row | right +-1 s | SFK % match |
|---|---|---|---|
| export as is | 39/54 | 49/64 | 45.5 |
| **keepers by goal + referee out (kept)** | **42/54** | **52/64** | **43.2** |
| + field players re-teamed by shirt | 44/54 | 53/64 | 38.8 |
| (all variants: windows 2/5/10/20 s, shirt/goal keeper rule, n1c / n1c2 / n1c3 reads in result.json) | 38-44 | 48-53 | 33-44 |

- Rows changed by the kept rule: referee out of the teams 6,498 rows (B 5,889 / A 609), keepers 4,472 rows; by eye on the repo frames
  `changed_*_to_ref.jpg` 43 of 46 are the referee (3 dark players), `changed_*_to_Ak/Bk.jpg` 26 of 28 are a keeper at that goal
  (2 orange keepers in the left goal, right by the goal rule). Possession rows moved: SFK -> AIK 369, to nobody (referee) 123.
- **Field players by shirt is NOT kept**: it moves the share most (38.8%) but by eye on the repo frames only about half of its
  changes are right (`field_changes/`: A->B ~40 right / ~25 dark players wrongly made white; B->A ~11 right / ~24 wrong, many of
  them whites) - small, blurred far shirts and foot points between two players. Its +2 moments are within noise of 54.
- Left: 12 misses at the nearest row, 10 of them AIK on the ball read SFK (white carriers read dark at that moment: q36, q32, p41).
  The whole-match share 43.2% is at the top of the by-eye range -> **AIK possession stays withheld** (Waiting for Daniel, N0-30b).

## Not done
Nothing in the pipeline or the app changed. The goal rule for keepers belongs in the export's keeper step, but on AIK it depends on
the attack direction Daniel has not settled (N0-26b: 'flip'); the referee filter needs the kit reader in the Kaggle relabel stage.
Both reach the app only with a re-join.

## Tests
`tests/test_n1c_kitread.py` (10: classes, ground ruler, grass check, frame seeking on a stand-in video) and
`tests/test_n1c_relabel.py` (5: decide, window vote follows an id slide, keeper by goal incl. the export's own flag, possession edit,
re-computation). Every runner job dry-run first on a local stand-in video.
