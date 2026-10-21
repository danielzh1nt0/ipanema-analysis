# N0-21b (21 Oct, worker, local $0, nothing on Modal)

**Card:** results/nightly/2026-10-21.md says nothing worse than 20 Oct; all 7 player checks ran (none NOT RUN), tests 147/147.
(No card by 04:20 UTC, so the backup nightly was pushed per the rules; it finished 04:45.)

**What was looked at:** a coverage run of the whole suite (513 pass) to find app code with no test. The app's
player loop `tracking.track` (detect -> ByteTrack -> team vote -> metres, lines 172-256) had none.

**New test:** `tests/test_track_loop.py` (3) - fake detector, fake kit model, the real supervision ByteTrack, a
45-frame synthetic video, 7 people:

| person reads | expected | result |
|---|---|---|
| always A / always B | team A / B | right |
| referee (R) | not a player | right |
| feet off the pitch (O) | not a player | right |
| A, A, 'neither' (cycled) | stays A | right |
| always 'neither' | K (clean() drops it later) | right |
| B, R, B, B (one referee read in 4) | stays B | right |

Also right: one stable id per person, feet 100,680 px -> 10,68 m through H, the kit check once per 3rd frame (+ new tracks).
**No fault in the loop's logic.**

**One small fault fixed:** `tracking.RAW_TEAM` (this frame's own colour reading per track, saved by tools/tracktest.py
as raw_team for the P6-style checks) was never cleared. When tracktest runs two detectors in one job
(DETECTORS=yolo,rfdetr) the second run could show the first run's readings for the same (frame, track id) - ByteTrack
ids start at 1 in both runs. Now cleared at the start of every `track()`. No saved result was affected: no committed
tracktest summary has both detectors in one job (default is rfdetr only). The app never reads it.

**Also checked, not changed:** `ball.bridge` fills ball gaps <= 1 s with a straight line in pixels, not metres. On both
app clips' exact picker inputs (SFK-BP 288 bridged frames, AIK 541) the camera moves only ~0.2 m (median, p90 0.5 m)
during a gap, so pan is not a problem; the two ways differ a lot only where an end of the gap projects wildly (ball in
the air), where the pixel line is the safer one. `ball.pick_confidence` and `ball.check` (the app's ball score) read
correct.

Full suite: 516 pass, 4 skipped.
