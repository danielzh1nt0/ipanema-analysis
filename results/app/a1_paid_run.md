# A1: the one paid run (prepared 5 Oct by the worker; nothing started)

Nothing below has run. Each line is one push; push only after Daniel says go in chat.
Demo rule: nothing is re-run after Monday noon, so this is for after the demo (Vallentuna is not in the demo library, so step 1 does not touch the demo).

| step | when | what | minutes (measured) | est. cost |
|---|---|---|---|---|
| 1 | do first | Vallentuna teams: apply the K3c team decisions (join only, CPU) | 25 (5 Oct K3b join: 17-24 min) | ~EUR 0.26 |
| 2 | do first | Vallentuna by-eye pictures after the join (CPU, cents) | 4 (5 Oct qa-frames: ~2-4 min) | ~EUR 0.02 |
| 3 | after the Lovable badge | SFK-BP re-join: 'needs review' flags in stats.json (CPU) | 15 (4 Oct night join: ~10-16 min) | ~EUR 0.16 |
| 4 | after the Lovable badge | SFK-AIK re-join: 'needs review' flags in stats.json (CPU) | 12 (4 Oct night join: ~5-12 min) | ~EUR 0.13 |
| 5 | after the Lovable badge | Fetch both new match_data/stats files to check them (CPU, cents) | 3 (fetch: ~1-3 min) | ~EUR 0.00 |
| 6 | optional, instead of 1 | Vallentuna clean fix: GPU re-track with the K3c rule, then join | 37 (5 Oct K3/K3b re-track: 36.7 min, ~78 GPU-min (10 L4s in parallel)) | ~EUR 1.51 |

Recommended now: steps 1-2, about EUR 0.28. After the Lovable 'needs review' badge: 3-5, about EUR 0.29.
Optional clean Vallentuna fix (6a, then 1-2): about EUR 1.79 by list prices; earlier notes said ~EUR 5 for a re-track, so take EUR 5 as the cap.
Costs = measured minutes x Modal list prices (not checked against the bill), +-50%.

## Why each step
- **1** App now shows nearly everyone as SFK (K3b, possession 79/21). The override puts ~3 of 4 players in the right team (app now ~half), owner check 13/17.
- **2** Check the 12 QA moments by eye before saying it is fixed.
- **3** Numbers should not move (same pieces); adds the V2 review flags and rewrites match_data (checks the goal 'B' vs 'A' found by V2).
- **4** Same as 3 for AIK (flags 10% of frames with 12+ of one team).
- **5** Confirms the SFK-BP goal team in match_data and reads the flags offline.
- **6** Cleaner than the override (each detection decided by the rule, no ids carried from the old export). Push 6a, wait for it to finish, then push 1 and 2.

## Exact pushes (from a clean, pulled main; one at a time, wait for 'finished' in results/live/<match>.log)
- 1:
```
echo "A1-1 Vallentuna join with the K3c team override, CPU only (Daniel's go) $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-1 Vallentuna join with the K3c team override, CPU only (Daniel's go) [full:p15u-vs-vallentuna-2026-10-03-6cce] [allow-fallback] [budget:40]" && git push origin main
```
- 2:
```
echo "A1-2 Vallentuna QA pictures + stats after the K3c join $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-2 Vallentuna QA pictures + stats after the K3c join [qa-frames:p15u-vs-vallentuna-2026-10-03-6cce] [fetch:runs/matches/p15u-vs-vallentuna-2026-10-03-6cce/stats.json]" && git push origin main
```
- 3:
```
echo "A1-3 SFK-BP re-join for the V2 review flags, CPU only (Daniel's go) $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-3 SFK-BP re-join for the V2 review flags, CPU only (Daniel's go) [full:SFKBP1109] [allow-fallback] [budget:30]" && git push origin main
```
- 4:
```
echo "A1-4 AIK re-join for the V2 review flags, CPU only (Daniel's go) $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-4 AIK re-join for the V2 review flags, CPU only (Daniel's go) [full:p15u-vs-aik-2026-09-21-bd09] [allow-fallback] [budget:30]" && git push origin main
```
- 5:
```
echo "A1-5 fetch the re-joined files $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-5 fetch the re-joined files [fetch:runs/matches/SFKBP1109/match_data.json] [fetch:runs/matches/SFKBP1109/stats.json] [fetch:runs/matches/p15u-vs-aik-2026-09-21-bd09/stats.json]" && git push origin main
```
- 6:
```
git mv overrides/p15u-vs-vallentuna-2026-10-03-6cce_teams.json overrides/p15u-vs-vallentuna-2026-10-03-6cce_teams.before_retrack.json && echo "A1-6a Vallentuna re-track with the K3c kit rule; team override moved aside (ids change) (Daniel's go) $(date -u)" >> triggers/last.txt && git add -A triggers overrides && git commit -m "A1-6a Vallentuna re-track with the K3c kit rule; team override moved aside (ids change) (Daniel's go) [full-rf:p15u-vs-vallentuna-2026-10-03-6cce] [retrack]" && git push origin main
```

## Not in this run (and why)
- F1 'full SFK-BP with the unsure-frames fix': already in - the 4 Oct joins ran with that code (stale waiting item).
- B4c spare-ball rule: Daniel's yes/no, off by default; would need its own re-join.
- K3b-only GPU re-track: superseded by 6a (K3c rule).
- D4 Bundesliga (~EUR 1-2), V1c shot clicks (1/40 clicked), S2 frame fetch, F1 AIK picker inputs: separate decisions, not app updates.

## Checks done
- tools/a1_paidrun.py: every tag exists in run-on-modal.yml, periods files present, joins carry a [budget:] stop, no stray '[run', re-track moves the id-keyed override aside.
- tests/test_a1_paidrun.py, tests/test_team_override.py pass locally.
