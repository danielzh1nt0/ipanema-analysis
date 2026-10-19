# N0-19c (19 Oct 08:20 UTC, worker, local, $0, nothing on Modal)

**Nightly card 19 Oct:** worse than 18 Oct = nothing. Tests 134/134 files; local suite 444 -> 449 pass, 4 skipped (no torch here).

No new main-session code since 04:20, so I re-ran the upload checks (V2) on the four match copies in the repo.

## Finding: the repo copy of SFK-BP (SFKBP1109) is not one export
- Its `stats.json` holds the check the export itself ran: **0%** of frames with 2+ keepers in a team, 22,007 player frames, ball off the pitch 19.8%.
- Re-running the same check on the frame files next to it gives **17.8%** frames with 2+ keepers (up to 7 "keepers" in one team, some in midfield), 21,807 player frames, ball off 19.3%.
- Same check, same code path -> the frame files come from older exports than `stats.json` (they were fetched one by one at different times). Per file: `sfkbp_chunks.md` - files 001 and 005 have no double keepers, the rest have hundreds.
- So the keeper flag on this copy is a false alarm about the copy, **not** an app fault. Anything that reads these SFK-BP full-match frame files (many tools/*lab.py) is reading a mix.
- AIK full match: only 2 of 12 frame files are in the repo, so its re-check numbers are not the app's either. Vallentuna: one whole export, numbers match the export's own check. SFK-BP 5-min clip: old export with no saved check, can't tell.

## Fix (checker only, app untouched)
`ipanema/uploadcheck.py` `copy_check`: `check_export` / `tools/upload_check.py` now say "COPY NOT ONE EXPORT - numbers below are not the app's" when
- frame files listed in `match_data.frame_chunks` are missing inside the match periods (missing ones outside the periods are fine), or
- the frame-only checks (players, keepers, ball) give different numbers than the export's own saved check.
Tests: `tests/test_upload_checks.py` +5 (whole copy OK, mixed file caught, missing file caught, missing file outside play OK, old export without a saved check not flagged).

Output on all four copies: `upload_checks.txt`.

## Not done (needs Modal)
A fresh copy of the SFK-BP frame files needs a `[fetch:]` run on Modal -> listed under Waiting for Daniel. Not urgent: the checker now warns.
