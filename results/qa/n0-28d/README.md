# N0-28d: why big clips stalled on the free runner (28 Oct, worker, local + free runner, $0, nothing on Modal)

## The cause
The runner installs `"numpy<2" opencv-python-headless`, which gives **OpenCV 4.11.0** (4.12+ needs numpy 2).
OpenCV 4.11.0 has a bug: the check for `OPENCV_FFMPEG_CAPTURE_OPTIONS` is upside down
(`cap_ffmpeg_impl.hpp` line 1140, `if(!options.empty())` skips the options). So our ffmpeg settings
(10-s read timeout + reconnect, `tools/clipopen.py`) were silently thrown away on the runner. Locally we
have OpenCV 5.0, where they work - that is why it passed here and not there. 4.10 and 4.12+ are fine.

## Proof
Same test server that goes silent on the header read and again part-way through a read:

| where | OpenCV | frames | time |
|---|---|---|---|
| this machine | 4.11.0 | 0 / 300, "moov atom not found" | 120 s (the runner's failure, exactly) |
| this machine | 4.10.0 | 300 / 300, reconnected twice | 10 s |
| free runner | 4.11.0 | 0 / 300 | 120 s |
| free runner | 4.10.0 (now pinned) | 300 / 300, reconnected twice (resumed at the exact byte) | 20 s |

Real R2 on the runner with 4.10, the nightly's 20-s piece from 25:00, 3 rounds:
Djursholm 3 / 3 and Spånga 3 / 3 opened first try, 599 / 599 frames, 8-13 s each.
(No stall happened during those 6 reads, so the stall proof is the test server above.)

## The fix
- `nightly.yml` (both installs) and `free.yml` pin `opencv-python-headless==4.10.0.84` (checked: rfdetr and
  supervision accept it, nothing pulls 4.11 back in).
- `tools/clipopen.py` logs a clear warning if it ever runs on 4.11 again; `tools/tracktest.py` logs the OpenCV version.
- Found in the dry run: when a stalled read left no output file, `read_piece` crashed; it now counts that try as 0 and retries.

The next nightly player numbers come from OpenCV 4.10 instead of 4.11 (same ffmpeg inside, so the frames should be
the same); if a row moves, that is the first thing to suspect.

## Files
`tools/n0_28d_runner_check.py` (the check), `runner_check.md/.json` (runner results),
`tests/test_n0_28d_cvpin.py` (7 tests, incl. a real stall-and-reconnect run). Full suite 737 pass (4 skipped).
