"""S6 (2 Oct): sequences / balls lost on the free Metrica pro data (2 games x 2 halves, clean + noisy) with the counted
stats on the raw possession state vs the S8 spell state (run.py default take 1.5 / join 3) and a few other spell
settings. Truth = Metrica's event chains (tools/metricalab.truth_sequences). Free, local.
    python tools/s6lab.py [--quick]   (quick = Game 1 half 1 only)"""
import sys, os, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import metricalab as ML
M = "/tmp/metrica/data"; quick = "--quick" in sys.argv; rows = []
NOISES = [a for a in sys.argv[1:] if not a.startswith("--")] or ["clean", "streak", "flicker", "flicker_heavy"]
NZ = {"clean": False, "streak": "streak", "flicker": "flicker", "flicker_heavy": "flicker"}
FL = (ML.FLICK_P, ML.FLICK_R, ML.FLICK_S); HEAVY = (0.1, 10.0, (0.3, 1.0))
SETS = [(0, 3.0), (1.0, 3.0), (1.5, 3.0), (1.5, 5.0), (2.0, 3.0), (3.0, 3.0)]
for g in (("Sample_Game_1",) if quick else ("Sample_Game_1", "Sample_Game_2")):
    d = f"{M}/{g}"; home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    frames = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; byper = {}
    for k, (p, _, _) in frames.items(): byper.setdefault(p, []).append(k)
    ev = ML.load_events(f"{d}/{g}_RawEventsData.csv"); T = ML.truth(ev, byper)
    for per in (sorted(T)[:1] if quick else sorted(T)):
        ts = ML.truth_sequences(ev, per); mins = len(byper[per]) / ML.FPS / 60
        for noisy in NOISES:
            ML.FLICK_P, ML.FLICK_R, ML.FLICK_S = HEAVY if noisy == "flicker_heavy" else FL
            for take, join in SETS:
                t0 = time.time(); O = ML.ours(frames, per, NZ[noisy], spell_take=take, spell_join=join)
                found, real = ML.seq_found(ts, O["sequence_starts"])
                rows.append({"game": g, "half": per, "noisy": noisy, "spell_take": take, "spell_join": join, "minutes": round(mins, 1),
                             "truth": len(ts), "ours": O["sequences"], "found": found, "real": real, "switches_raw": O["switches_per_min_raw"], "switches": O["switches_per_min"],
                             "lost_truth": sum(T[per]["balls_lost"].values()), "lost_ours": sum(O["balls_lost"].values()),
                             "passes_truth": sum(T[per]["passes"].values()), "passes_ours": sum(O["passes"].values())})
                print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
out = "results/possession/s6lab_quick.json" if quick else "results/possession/s6lab_2026-10-02.json"
if os.path.exists(out) and not quick:                                       # keep rows of noises not re-run
    rows = [r for r in json.load(open(out)) if r["noisy"] not in NOISES] + rows
json.dump(rows, open(out, "w"), indent=1)
agg = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    a = agg[(r["noisy"], r["spell_take"], r["spell_join"])]
    for k in ("truth", "ours", "found", "real", "lost_truth", "lost_ours", "passes_truth", "passes_ours", "minutes"): a[k] += r[k]
print("noisy take join | seq truth ours found% real% | per team per 90 | lost truth ours | passes truth ours")
for (n, tk, jn), a in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
    print(f"{str(n):5} {tk:4} {jn:4} | {a['truth']:4} {a['ours']:4} {100*a['found']/a['truth']:3.0f}% {100*a['real']/max(1,a['ours']):3.0f}% | "
          f"{a['ours']/a['minutes']*90/2:4.0f} (truth {a['truth']/a['minutes']*90/2:4.0f}) | {a['lost_truth']:4} {a['lost_ours']:4} | {a['passes_truth']:4} {a['passes_ours']:4}")
