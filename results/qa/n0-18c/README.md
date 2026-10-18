# N0 18 Oct 16:20 - the nightly 'Tests' row could not see a failing test

Card (results/nightly/2026-10-18.md): nothing worse than 17 Oct. No new main-session code since 12:20, so the worker
checked the nightly itself (`tools/nightly_quick.py`, which makes the card's "Tests 129/129 pass" row).

Found and fixed:
- **A failing test was counted as passing.** When pytest failed on a test file, the nightly re-ran that file as a plain
  script and believed the script's answer. 130 of 131 test files are pytest-style: run as a script they only define
  their tests and exit 0 without running any. So almost any test failure was hidden from the card.
  It was hiding one every night: `tests/test_ballscorer.py` needs torch, the nightly's quick job has no torch, pytest
  failed, the script re-run "passed". Now only a script-style test (own `__main__` block, 1 file) gets the second try;
  everything else that pytest fails is listed as failed. `test_ballscorer` now skips cleanly without torch instead.
- The nightly ran at import; it now runs from `main()`, so it can be tested.
- tests/test_nightly_quick.py (5 tests, real pytest runs on made-up passing/failing/script-style files, plus fakes).

Dry run of the whole nightly quick job here (same packages as the runner, no torch): tests 132 files, 0 failed;
ball 22/34 (best 25); pro-data stats identical to the committed 18 Oct numbers row for row (passes 15% / possession
8 pts off, sequences 40% / 44%). Every test file was also run one by one under pytest: the only one pytest failed was
test_ballscorer (torch), so no other failure was being hidden today. The card's test count will read 132/132 from
tomorrow (2 new test files today + this one); first real check is the 03:00 nightly.

Suite: 419 pass, 4 skipped. $0, nothing on Modal, no free-runner push (the nightly picks it up on its own).
