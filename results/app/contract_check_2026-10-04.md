# Contract check of the clean first-half exports (4 Oct, tools/contract_check.py)

Both SFK-BP and SFK-AIK first halves (re-exported 4 Oct, in the app):
- every event carries `tier` (SFK-BP: 7 verified, 304 beta, 525 hidden; AIK: 16 / 324 / 549) and `verified`; verified = goals + shots only, beta = turnovers only
- goals and shots all inside the period (+-5 s); no phantom goal at 0:28; goal counts agree between events and `metrics.shots`
- set pieces: `stoppage` / `throw-in` / `corner` / `goal kick` (no "free kick")
- turnover payloads carry `pressed_within_2s`, `near_at_2s`, `time_to_press`, `t_won` / `forward_within_3s`, `lost_back_5s`
- frames carry `pressed` and it equals `pressure_m <= 2` (AIK chunk 3: 724 carrier frames checked)
- team rows: possession sums to 100, losses(A) = recoveries(B)
- one correction to the contract doc: players have `state` "observed" (97.5%) or "filled" (short gap bridged, conf 0.5)

(SFK-BP frame chunks on this machine are from the 3 Oct export; the 4 Oct chunk check was done on AIK.)
