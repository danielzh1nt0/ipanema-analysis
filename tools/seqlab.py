"""S6 (2 Oct): sequence rule (take_s, join_s) on the free Metrica pro data, 2 games x 2 halves, clean + noisy: true
sequences found (start within 2 s, same team) and share of ours that are real; plus the SFK-BP clip count. Free, local."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "tools")
import metricalab as ML
from ipanema import possession as P
M = "/tmp/metrica/data"; rows = []
for g in ("Sample_Game_1", "Sample_Game_2"):
    d = f"{M}/{g}"; home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; byper = {}
    for k, (p, _, _) in frames.items(): byper.setdefault(p, []).append(k)
    ev = ML.load_events(f"{d}/{g}_RawEventsData.csv"); T = ML.truth(ev, byper)
    for per in sorted(T):
        for noisy in (False, True):
            for take, join in ((0.5, 3.0), (1.0, 3.0), (1.0, 5.0), (1.5, 5.0), (2.0, 6.0)):
                P.SEQ_TAKE_S, P.SEQ_JOIN_S = take, join
                O = ML.ours(frames, per, noisy)
                ts = ML.truth_sequences(ev, per); found, real = ML.seq_found(ts, O["sequence_starts"])
                rows.append({"game": g, "half": per, "noisy": noisy, "take_s": take, "join_s": join, "truth": len(ts), "ours": O["sequences"], "found": found, "real": real}); print(rows[-1], flush=True)
json.dump(rows, open("results/possession/seqlab_2026-10-02.json", "w"), indent=1)
import collections
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in rows:
    a = agg[(r["take_s"], r["join_s"])]; a[0] += r["truth"]; a[1] += r["ours"]; a[2] += r["found"] or 0; a[3] += r["real"] or 0
for k, (t, o, f, r) in agg.items(): print(f"take {k[0]} join {k[1]}: truth {t} ours {o} found {100*f/t:.0f}% real {100*r/max(1,o):.0f}%")
