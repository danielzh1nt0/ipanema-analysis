"""N0-24c summary: per ball noise x rule, totals over the 4 Metrica halves (noise rows = 2 seeds)
    PYTHONPATH=. python tools/n0_24c_summary.py [results/qa/n0-24c/metrica_entries.json]"""
import sys, json
from collections import defaultdict

def summary(rows):
    agg = defaultdict(lambda: defaultdict(int)); err = defaultdict(list)
    for r in rows:
        key = (r["noise"], r["rule"]); m = r["matched"]
        for f in ("truth", "ours", "found", "how_same", "truth_pass", "found_pass"): agg[key][f] += m[f]
        for tm in ("A", "B"):
            t = r["teams"][tm]; agg[key]["ours_pass"] += t["ours_pass"]
            err[key].append(100 * (t["ours"] - t["truth"]) / max(1, t["truth"]))
    out = []
    for key, a in agg.items():
        e = err[key]
        out.append({"noise": key[0], "rule": key[1], **a, "found_pct": round(100 * a["found"] / a["truth"]), "real_pct": round(100 * a["found"] / max(1, a["ours"])),
                    "count_ratio": round(a["ours"] / a["truth"], 2), "team_err_pct_min": round(min(e)), "team_err_pct_max": round(max(e)),
                    "pass_found_pct": round(100 * a["found_pass"] / max(1, a["truth_pass"])), "how_same_pct": round(100 * a["how_same"] / max(1, a["found"]))})
    return out

if __name__ == "__main__":
    rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/qa/n0-24c/metrica_entries.json"))
    print("noise    rule   truth ours found% real% ours/truth team err% pass-entries found% how right%")
    for s in summary(rows):
        print(f"{s['noise']:8} {s['rule']:6} {s['truth']:5} {s['ours']:4} {s['found_pct']:6} {s['real_pct']:5} {s['count_ratio']:10} {s['team_err_pct_min']:5}..{s['team_err_pct_max']:<4} {s['pass_found_pct']:19} {s['how_same_pct']:5}")
