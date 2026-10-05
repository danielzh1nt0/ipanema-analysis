# V2 – automatic checks on every upload (5 Oct, worker, local, $0)

Every export now runs 6 checks (`ipanema/uploadcheck.py`, called from `export.write`). Parts that fail are listed in
`stats.json → review.needs_review` (also in `summary.json`), with the numbers in `review.checks`. The app should then say
"needs review" for that part instead of showing it. A failing check never stops an upload.

| check | flags | rule |
|---|---|---|
| players per team per frame | `players` | median observed per team >= 4, smaller team >= half the bigger, > 11 of one team in <= 5% of frames |
| keepers | `keepers` | 2+ keepers in one team in <= 2% of frames |
| goals | `goals` | goal (as the app counts it, stats.json) at the end that team attacks; the OTHER team kicks off after it (ball held on the centre spot, read from the frames); goal events in match_data say the same as stats.json. A kick-off reading never overrules a team checked by eye. |
| shots located | `shot_map` | >= 80% of shots have x/y |
| ball off the pitch | `ball_map` | <= 10% of ball positions (metres, the mini-map) more than 3 m outside the lines |
| possession vs by-eye key | `possession` | >= 70% of clear by-eye moments right (needs >= 10 moments; keys: `reference/<match>/owner_key.json` or the `owner` field of `ball_key_graded.json`) |

New key: `reference/SFKBP1109/owner_key.json` = the 58 clear who-has-the-ball moments of 28 Sep moved to match time
(clip t + 1200 s, dark = A). Checked: this way round agrees 44/58 with the full-half export, the other way round 4/58.

## On the three matches as the repo has them (`PYTHONPATH=. python tools/upload_check.py <id>`)

Caveat: most frame files in the repo are from BEFORE the 4 Oct night re-run (12 QA moments: player counts differ from the
server's on 8 of 12 Vallentuna and 8 of 12 SFK-BP). Refreshing them needs a Modal fetch, so not done.

| | SFK-BP | AIK | Vallentuna |
|---|---|---|---|
| players | OK 7 / 7 | **FLAG** 6 / 6, but 10% of frames have 12-16 of one team (one sample frame: 4 of 12 standing on the near touchline) | OK 5 / 7 |
| keepers | **FLAG** 18% of frames (old files: the goalmouth bug fixed 4 Oct night) | **FLAG** 2.7% | **FLAG** 9% (old files); the 3 re-fetched files: 0 of 3,814 frames |
| goals | **FLAG** stats.json says 2619 goal A (by eye), the repo's match_data event says B (app timeline would show GOAL · B); B kicks off after = A scored | ?? kick-offs not in the fetched frames | OK - kick-off by the other team after 4 of 5 goals; 3840 our possession says the scorer, team by eye kept |
| shot map | **FLAG** 0/7 located | **FLAG** 0/16 | **FLAG** 0/17 (V1 is adding them by eye) |
| ball map | **FLAG** 19% off the pitch | OK 1.4% | **FLAG** 10.4% (re-fetched files 8.6%) |
| possession | OK 44/58 (76%) | ?? no key | **FLAG** 10/17 (59%; matches the 52% in reference/<vall>/README.md, the app already withholds it) |

By eye (results/qa/full/SFKBP1109, server pictures): the ball ring is on the ball at the 3 moments where the repo files put
the ball off the pitch, but the metres are wrong - a ball on the goal line read 5 m behind it (m00), a ball the keeper holds
read 14 m behind the goal (m10), a ball in the air above the trees read 1 km away (m05). So the flag is right: the video ring
is fine, the mini-map dot is not.

The kick-off reading was checked against the 6 goals whose team is known by eye: 5 right, 1 wrong (Vallentuna 3840,
uncalibrated frames). That is why it can only flag a goal whose team was NOT checked by eye.

## For the app (Lovable, waits for Daniel)
"Read `stats.json → review.needs_review` (list of: players, keepers, goals, shot_map, ball_map, possession). For each part in
the list show a small grey 'Needs review' badge on that card / layer and don't draw its numbers or markers. Missing
`review` = nothing to flag."

Files: `results/qa/v2/<match>.json` (full check output), `results/qa/v2/vallentuna_postfix_chunks.json`.
Tests: `tests/test_upload_checks.py` (10, incl. an export dry run).
