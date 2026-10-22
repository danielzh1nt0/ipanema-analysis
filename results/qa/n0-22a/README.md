# N0-22a (22 Oct 00:50 UTC, worker, local, $0, nothing on Modal)

**Nightly card 21 Oct:** worse than 20 Oct = nothing. No 22 Oct card yet (scheduled run not started).

**Queue before this:** M2 (main session), PC2 (waits for M2), BALL-EDGE (needs a Kaggle GPU; this worker runs only local + the free runner) - skipped, left open.

## Finding
Full test suite here, in a Python **without torch** (like the nightly runner): **8 failed**, all in `tests/test_ballrf_edge.py` (added yesterday, N0-21e). Cause: `ballrf.autocast()` and `ballrf._fp16()` did `import torch` before checking whether fp16 is on at all. Yesterday's worker had torch installed, so it never showed. Tonight's card would have listed `test_ballrf_edge.py` under "Worse than 21 Oct".

## Fix
`ipanema/ballrf.py`: torch is imported only when `IPANEMA_RF_FP16=1`. With it on, behaviour is exactly as before. Nothing in the app changes (the app always has torch).

## Test
`tests/test_no_torch.py` (4): autocast/_fp16 are no-ops without torch when off; with it on they still ask torch; the stand-in ball finder runs without torch; and `test_ballrf_edge.py` run in a child Python where `import torch` fails passes. On the old code 3 of the 4 fail.

Suite here (no torch): 8 failed / 528 passed -> **540 passed, 0 failed**.
