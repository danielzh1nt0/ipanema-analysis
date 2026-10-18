# N0-18d (18 Oct 20:20 UTC, worker, local $0, nothing on Modal, no free-runner push)

Card: results/nightly/2026-10-18.md says **nothing worse** than 17 Oct.

New main-session code since the last worker run: the data-strategy commit 0d0c0da (tools/reviews_to_labels.py, a new
step in the nightly 'quick' job, coach grading in tools/export_qa.py). It had no tests. Checked and tested it:

## Faults found and fixed
- **Coach grading broke after any re-run.** export_qa counted a coach-confirmed event as "still present" only when the
  exact event id was in the export. Ids move on a re-run: `pass_<t>_<player id>` (player ids change with re-tracking),
  `lost_<frame>`. So every confirmed pass would have read as lost after the next re-run. Now: same id, or the same kind
  of event (id prefix) within 1.5 s of the reviewed time.
- **Wrong event type when the event is no longer in the export** (or match_data.json in storage can't be read - it is
  not public, see F1): the type was the id prefix (`lost`, `hto`, `seq`), not the export's type (`turnover_lost`,
  `high_turnover`, `sequence_end`), so kick windows mixed two names. Now mapped; the time is also read from the id
  where the id carries seconds (pass, restart, hto, shot).
- **Two corrections of one field**: the app returns rows in no set order, so a random one won. Now the latest
  (updated_at / created_at) wins.
- Kick windows could start before 0 s (event in the first second). Clamped.
- results/labels is now created if missing (the summary write would have crashed); the export is only downloaded for
  matches that have reviews.

## Checked, no fault
- The new nightly step under the runner's `bash -e`, dry run in a throw-away repo with a local remote:
  without the Supabase secrets (as tonight - D5) it prints "not set", commits nothing, step passes;
  with stand-in secrets + a fake Supabase it writes the keys, commits and pushes. Tonight's nightly will not break.

## Tests
tests/test_reviews_to_labels.py (8, fake Supabase, no secrets/network) incl. a check on the real SFK-BP export that every
event finds itself; 6 of them fail on the old code. The app's real table columns could not be checked here
(app repo not in this session's scope) - the tool's own column names are assumed.
