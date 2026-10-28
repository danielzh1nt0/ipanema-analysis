# N0-28c: the nightly player rows were frozen since 28 Sep (28 Oct, worker, free runner + local, $0, nothing on Modal)

## What started it
The 28 Oct card (scheduled run, 05:53) said: `Worse than 2026-10-27 | p15u-vs-djursholm-2026-09-26: player check did not run (job failed)`.

## What was wrong (3 faults, all fixed)
1. **One R2 stall lost the Djursholm check.** OpenCV waited 30 s for the video header, R2 stopped sending
   ("Stream ends prematurely"), 0 frames, job failed. The same file opened fine 25 min earlier.
   Fix: `tools/clipopen.py` opens the clip up to 3 times (waits 20 s, 40 s) and re-reads the 20-s piece if the
   stream dies part-way. Re-run: Djursholm ran normally.
2. **4 of the 5 player rows on the card were old numbers - from 28 Sep.** Each player job uploads every
   `results/qa/tracktest*` folder of its checkout, and the card step copied all of them on top of each other,
   so the last job copied (Solberga) put back the repo's old copies of the other four. Only Solberga's row was ever
   live; SFK-BP, Djursholm, Reymersholm and Spånga had not changed since 28 Sep (GitHub file history), so a drop
   there could never show as 'worse'. Fix: `nightly_scorecard.py --collect` takes from each job only its own
   match's folder (tested with the real folder layout: each row now shows its own job's numbers; the old step
   reproduces the bug - only Solberga changes).
3. **The SFK-BP check never really ran.** Over the R2 stream it read 2-23 of the 599 frames on all 4 runs
   looked at today (OpenCV reads the 300 MB file from the start to the 60-s mark and R2 cuts it after ~70 MB),
   tracked ~0.5 s and reported 0 players per frame - as a success. Nobody saw it because of fault 2.
   Fix: a piece shorter than 90% now fails the job (card says NOT RUN, not wrong numbers), and clips under 1.5 GB
   (SFK-BP) are downloaded whole first (curl with retries); big full-match files still stream, which works.

4. **R2 stalls are common, not rare.** The 3rd re-run lost Spånga: 3 opens in 2.5 min all hit a 30-s stall at the
   header. Reproduced locally with a test server that goes silent mid-file: plain OpenCV gets 0 of 100 frames (same
   'count -1, 30 s' as the runner). Fix: for URLs ffmpeg's own HTTP options (10-s read timeout + reconnect at the same
   byte) and a 2-min OpenCV budget: locally 100/100 frames through 1 stall and through 3 stalls (header or mid-file);
   the whole-file download also gives up a stalled try after 20 s and retries.
   **On the runner it did not help yet:** the 4th re-run (10:43) got SFK-BP (download), Spånga, Reymersholm and
   Solberga, but Djursholm failed 3 opens - each sat silent for the whole 2 min with no 'reconnect' in the log, so
   the runner's OpenCV build seems to ignore the ffmpeg options (local OpenCV is 5.0; the runner installs an older
   build with numpy<2). Left as N0-28d. The job now fails honestly (NOT RUN) instead of showing wrong numbers.

Also: the progress push inside `tracktest.py` failed on `git pull --rebase` (no committer name); fixed.

## Fresh numbers vs the frozen 28 Sep ones (re-runs 28 Oct 09:47-11:06)
| match | 28 Sep (what the card showed) | today, fresh |
|---|---|---|
| Djursholm | dark 6 / light 8 per frame, tracked 1.62 / 2.4 s | dark 8 / light 8, tracked 13.41 / 1.63 s |
| Reymersholm | 6 / 4, 1.79 / 2.54 s | 5 / 4, 2.02 / 2.8 s |
| Spånga | 7 / 6, 2.44 / 2.54 s | 8 / 7, 3.52 / 4.8 s |
| SFK-BP | 8 / 7, 7.81 / 9.78 s | 7 / 7, 7.82 / 9.38 s (3rd and 4th re-run, downloaded whole) |

By eye (Djursholm, compare_0085 / compare_0341): every red and white player on the pitch is ringed with the right
colour, nobody behind the fence; the referee is left out on 0085; one neon-yellow figure (keeper or referee) is
ringed white on 0341. The frozen 28 Sep picture of the same frame had 4 red rings on spectators behind the far goal
and missed most red players - so the month's work shows here, the card just couldn't see it.
The 'worse' lines on today's card (Djursholm light tracked 2.4 -> 1.63 s, Reymersholm dark 6 -> 5) compare against
28 Sep, not yesterday; from tomorrow the comparison is night to night.

## Tests
`tests/test_n0_28c_clipopen.py` (14): retries, cut piece, longest read kept, download / stream choice, collect
takes only each job's own folder, a failed job's folder is left alone, the workflow uses `--collect`.
Full suite 729 pass (4 skipped). Dry runs before each push: a local mp4, a dead URL (3 tries then exit 1), and an
HTTP server with range requests (stream and download paths).
