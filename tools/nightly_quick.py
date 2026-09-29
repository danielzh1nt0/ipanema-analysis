"""nightly: the quick free checks (tests, ball picker on the 34 moments, stats on Metrica, kits on 8 matches) -> results/nightly/quick.json"""
import subprocess, sys, json, os, glob, re, time
out = {"date": time.strftime("%Y-%m-%d")}
fails = []
for t in sorted(glob.glob("tests/test_*.py")):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", t], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    if r.returncode not in (0, 5):
        r2 = subprocess.run([sys.executable, t], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
        if r2.returncode: fails.append(os.path.basename(t))
out["tests"] = {"total": len(glob.glob("tests/test_*.py")), "failed": fails}
r = subprocess.run([sys.executable, "tools/picklab.py"], capture_output=True, text=True)
m = re.search(r"v2, new players\s+(\d+)/(\d+) right \(best possible (\d+)\)", r.stdout)
out["ball_34_moments"] = {"right": int(m.group(1)), "of": int(m.group(2)), "best_possible": int(m.group(3))} if m else {"error": (r.stdout + r.stderr)[-500:]}
M = "/tmp/metrica/data"
if not os.path.exists(M): subprocess.run("git clone -q --depth 1 https://github.com/metrica-sports/sample-data.git /tmp/metrica", shell=True)
sys.path.insert(0, "tools"); import metricalab as ML
errs = []
for g in ("Sample_Game_1", "Sample_Game_2"):
    d = f"{M}/{g}"; home = ML.load_tracking(f"{d}/{g}_RawTrackingData_Home_Team.csv", 100); away = ML.load_tracking(f"{d}/{g}_RawTrackingData_Away_Team.csv", 200)
    fr = {k: (home[k][0], home[k][1] + away[k][1], home[k][2]) for k in home}; bp = {}
    for k, (p, _, _) in fr.items(): bp.setdefault(p, []).append(k)
    T = ML.truth(ML.load_events(f"{d}/{g}_RawEventsData.csv"), bp)
    for per in (1, 2):
        for noisy in (False, "streak"):
            O = ML.ours(fr, per, noisy); t, o = T[per]["passes"], O["passes"]
            ts = ML.truth_sequences(ML.load_events(f"{d}/{g}_RawEventsData.csv"), per); sf, sr = ML.seq_found(ts, O["sequence_starts"])
            errs.append({"game": g[-1], "half": per, "noise": str(noisy), "pass_err_pct": round(100 * max(abs(o[s] - t[s]) / t[s] for s in t)),
                         "poss_err_pts": abs(O["possession_pct"]["Home"] - T[per]["possession_pct"]["Home"]),
                         "seq_truth": len(ts), "seq_ours": O["sequences"], "seq_found": sf, "seq_real": sr, "mode": O["mode"]})
out["stats_pro_data"] = {"worst_pass_err_pct": max(e["pass_err_pct"] for e in errs), "worst_poss_err_pts": max(e["poss_err_pts"] for e in errs),
                         "possession_model": errs[0]["mode"],   # E5 (29 Sep): 'simple' = the pipeline default; before 29 Sep the nightly scored 'viterbi'
                         "sequences_found_pct": round(100 * sum(e["seq_found"] for e in errs) / max(1, sum(e["seq_truth"] for e in errs))),
                         "sequences_real_pct": round(100 * sum(e["seq_real"] for e in errs) / max(1, sum(e["seq_ours"] for e in errs))), "rows": errs}
os.makedirs("results/nightly", exist_ok=True); json.dump(out, open("results/nightly/quick.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "stats_pro_data"}, indent=1), out["stats_pro_data"]["worst_pass_err_pct"])
