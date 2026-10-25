"""N0-25a: table of tools/n0_25a_runs.py results (both Metrica games, 2 seeds each, mean) -> results/qa/n0-25a/summary.json"""
import json, sys, numpy as np
out = {}
for sw in ("swap0.0", "swap0.25"):
    rows = {}
    for g in (1, 2):
        R = json.load(open(f"results/qa/n0-25a/metrica_game{g}.json"))["results"]
        for s in (0, 1):
            r = R[f"{sw}_seed{s}"]
            for rule, v in r.items():
                if not isinstance(v, dict): continue
                x = rows.setdefault(rule, {"truth": 0, "ours": 0, "found": 0.0, "real": [], "n": 0})
                x["truth"] += r["truth_sprints"]; x["ours"] += v["ours"]; x["found"] += v["found_pct"] * r["truth_sprints"] / 100
                if v["real_pct"] is not None: x["real"].append((v["real_pct"], v["ours"]))
    out[sw] = {rule: {"ours_vs_truth_pct": round(100 * x["ours"] / x["truth"]), "found_pct": round(100 * x["found"] / x["truth"]),
                      "real_pct": round(sum(p * n for p, n in x["real"]) / max(1, sum(n for _, n in x["real"]))) if x["real"] else None} for rule, x in rows.items()}
json.dump(out, open("results/qa/n0-25a/summary.json", "w"), indent=1)
for sw, t in out.items():
    print(sw); [print(f"  {r:16s} ours/truth {v['ours_vs_truth_pct']:4d}%  found {v['found_pct']:3d}%  real {v['real_pct']}%") for r, v in t.items()]
