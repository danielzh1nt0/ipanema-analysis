# N0-20c (20 Oct 12:20 UTC, worker, local, $0, nothing on Modal)

**Nightly card 20 Oct:** worse than 19 Oct = nothing (tests 140/140, ball 22/34, pro-data passes 15% / possession 8 pts off, all 7 player rows run; the history file has had the same ball/pass/possession numbers since 5 Oct).

**Full local test suite + coverage** of the app code: after this change 488 pass, 4 skipped (no torch / pyarrow here). The 4 that first failed were this machine missing `supervision` (3) and one slow test timing out under coverage; all pass after installing it. Covered: analytics 99%, metrics 99%, veo 99%, kits 98%, periods 98%, uploadcheck 97%, possession 95%, export 87%. The one export part no test called was the **pass network** (`metrics.pass_network`, in every export's `stats.metrics`).

## Finding: pass-network nodes sat in the wrong place or were missing
- A node was the track's whole-match average spot from `stats.players`.
- A track too short for the players' list had **no node**, so its edges pointed at nothing: SFK-BP clip 2, SFK-BP half 6, AIK 12, Vallentuna 9 edges.
- A track that swapped team (ID slide, the M2 problem) put an A pass on a B player's average spot: 4-37 edges per team per match touch such a track. Those nodes moved a median 9-22 m with the fix.

## Fix (on by default, `IPANEMA_NET_NODES=avg` = old)
A node now sits where that player played and received **this team's** completed passes (the usual way pass networks are drawn), with `n` = passes played + received. This matches the app's own pass map, which uses the same pass spots. Edges are unchanged. Result: 0 edges without a node in all 4 repo matches. Typical node move is 7-10 m, because a whole-match average pulls everyone towards the middle.

Picture: `network_SFKBP1109_s1200.jpg`, top = old, bottom = new; red rings = edges to a player with no node. By eye, the new nodes spread over the whole width. The dark keeper's goal kicks now come from one node on the goal line, where the old network had them near the 18 m line. Numbers: `network_compare.json` (`tools/n0_20c_network.py`).

Still true: nodes are track pieces, not named players (40 edges for 41 passes on the clip). The network only means something after M2. This fix makes it consistent, not meaningful yet.

Tests: `tests/test_pass_network.py` (7: completed-only edges per team, node = mean pass/reception spot, missing track gets a node, other-team track placed by this team's passes, no crash without spots/team, `avg` reproduces the repo export exactly, default/env switch). Reaches the app at the next run.
